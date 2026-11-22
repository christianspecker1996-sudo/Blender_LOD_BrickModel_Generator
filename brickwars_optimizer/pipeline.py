"""Run the rebuild in cancellable stages and clean up temporary data."""
from pathlib import Path
import tempfile
from types import SimpleNamespace
from uuid import uuid4

import bpy
import numpy as np
from mathutils import Matrix, Vector

from . import atlas, mesh, parts, rebuild, studs, uv


def selected(context):
    return [obj for obj in context.selected_objects if obj.type == 'MESH']


def collection(scene, name):
    result = bpy.data.collections.new(name)
    scene.collection.children.link(result)
    return result


def join_into(context, result, parts, temporary, origin):
    """Join copies of parts into result (all already in the shared snapshot space)."""
    copies = []
    for part in parts:
        copy = bpy.data.objects.new("BrickWars Join Part", part.data.copy())
        temporary.objects.link(copy)
        copy.location = origin
        copies.append(copy)
    if copies:
        meshes = [c.data for c in copies]
        mesh.activate(context, result, copies)
        bpy.ops.object.join()
        for data in meshes:
            if not data.users:
                bpy.data.meshes.remove(data)


RAY_FLAGS = ("visible_camera", "visible_diffuse", "visible_glossy", "visible_transmission",
             "visible_volume_scatter", "visible_shadow")


def bake_lighting(context, result, sources, material, diffuse, reach, margin, extrusion,
                  strength, size=None, bevel=0.0, tangent=None, highlight=0.0,
                  look='STUDIO'):
    """Bake ambient occlusion and source normals, then shade the diffuse atlas.

    Returns the occlusion per texel for the specular mask."""
    scene = context.scene
    size = size or max(max(result.dimensions), 1e-6)
    hidden = {}
    for obj in scene.objects:
        if obj not in sources:
            hidden[obj] = [getattr(obj, flag) for flag in RAY_FLAGS]
            for flag in RAY_FLAGS:
                setattr(obj, flag, False)
    world = scene.world
    created_world = world is None
    if created_world:
        world = scene.world = bpy.data.worlds.new("BrickWars Bake World")
    distance = world.light_settings.distance
    world.light_settings.distance = size * .06
    temporary = []
    try:
        for kind in ('AO', 'OBJECT'):
            target = atlas.image("BrickWars " + kind, diffuse.size[0], True)
            temporary.append(target)
            node = atlas.bake(context, result, sources, material, target, True, reach, margin,
                              extrusion, kind=kind, samples=128 if kind == 'AO' else 1,
                              bevel=bevel)
            material.node_tree.nodes.remove(node)
        return atlas.apply_lighting(diffuse, temporary[0], temporary[1], strength,
                                    tangent, highlight, look)
    finally:
        world.light_settings.distance = distance
        if created_world:
            scene.world = None
            bpy.data.worlds.remove(world)
        for obj, values in hidden.items():
            for flag, value in zip(RAY_FLAGS, values):
                setattr(obj, flag, value)
        for image in temporary:
            bpy.data.images.remove(image)


def run(context, settings):
    originals = selected(context)
    if not originals:
        raise ValueError("Select at least one mesh object.")
    selection = list(context.selected_objects)
    active = context.view_layer.objects.active
    primary = active if active in originals else originals[0]
    name = primary.name
    origin = primary.matrix_world.translation.copy()
    guided = settings.method == 'GUIDED'
    per_part = settings.method == 'PARTS'
    if guided and (active not in originals or len(originals) < 2):
        raise ValueError("Select the rebuilt meshes and make the high-poly source active.")
    if settings.bake and not settings.generate_uv:
        raise ValueError("Enable Generate UV Map to bake the textures.")
    source_objects = [primary] if guided else originals

    output = temporary = result = material = directory = None
    stud_images, stud_materials = [], []
    images = []
    complete = False
    stats = {}
    pieces = []
    try:
        yield .02, "Reading evaluated meshes"
        output = collection(context.scene, name + " Low Poly")
        temporary = collection(context.scene, "BrickWars Bake Sources")
        sources = [mesh.snapshot(context, obj, temporary, origin) for obj in source_objects]
        before = sum(mesh.triangles(obj.data) for obj in sources)
        if not before:
            raise ValueError("The selection contains no faces.")

        # Studs are found on the sources and taken off before fitting and baking.
        # For sprites, visible studs are merged into lattice panels, and the panels
        # get their share of the triangle budget.
        budget, found, chosen, ratio = settings.target, [], [], studs.PITCH_RATIO
        if settings.studs != 'TEXTURE':
            yield .03, "Finding studs"
            for number, source in enumerate(sources):
                for stud in studs.detect(source.data):
                    stud.source = number
                    found.append(stud)
            if settings.studs == 'SPRITES' and found:
                studs.score(found, sources, studs.extent(sources))
                cap = float('inf') if guided else settings.target * .5
                chosen, ratio = studs.choose(found, cap)
                budget = max(24, settings.target - studs.TRIS_PER_STUD * len(chosen))

        # Shapes are fitted without studs (slopes and profiles would enclose them),
        # but Rebuild Parts still culls hidden faces against the parts with studs.
        occluders = None
        if found:
            if per_part and settings.cull_hidden:
                occluders = []
                for source in sources:
                    copy = source.data.copy()
                    copy.calc_loop_triangles()
                    occluders.append(SimpleNamespace(data=copy))
            for number, source in enumerate(sources):
                studs.remove(source.data, [s for s in found if s.source == number])

        if guided:
            yield .35, "Assembling the rebuilt parts"
            guides = [mesh.snapshot(context, obj, temporary, origin)
                      for obj in originals if obj != primary]
            result = bpy.data.objects.new(name + "_LOWPOLY", guides[0].data.copy())
            output.objects.link(result)
            result.location = origin
            join_into(context, result, guides[1:], temporary, origin)
            mesh.triangulate(result.data)
            context.view_layer.update()
            fallback = [rebuild.material_color(result.data.materials[f.material_index]
                                               if f.material_index < len(result.data.materials)
                                               else None)
                        for f in result.data.polygons]
            extrusion = max(1e-6, max(result.dimensions) * .035)
            reach = 2 * extrusion
        elif per_part:
            share = settings.share_textures and settings.generate_uv
            task = parts.rebuild_parts(sources, budget, settings.cull_hidden,
                                       share, upright=settings.lighting > 0,
                                       occluders=occluders)
            try:
                while True:
                    try:
                        fraction = next(task)
                    except StopIteration as finished:
                        data, fits, keep, stats = finished.value
                        break
                    yield .05 + .4 * fraction, ("Finding identical parts" if fraction < .1 else
                                               "Fitting part shapes" if fraction < .75
                                               else "Removing hidden faces")
            finally:
                for blocker in occluders or ():
                    bpy.data.meshes.remove(blocker.data)
            # Removed studs still set the scale of rounded edges.
            stats["protrusion"] = max([stats["protrusion"]] + [s.height for s in found])
            # Sprites only stand on visible rebuilt shapes; omitted parts (texture only)
            # and hidden parts drop theirs.
            index = {old: new for new, old in enumerate(keep)}
            present = np.empty(len(data.polygons), dtype=np.int64)
            data.attributes["brickwars_part"].data.foreach_get("value", present)
            present = set(present.tolist())

            def fitted(stud):
                number = index.get(stud.source)
                return (number in present and fits[number].kind != 'omit'
                        and studs.on_shape(stud, fits[number].normals, fits[number].offsets))

            # Dropping a stud can split a panel; never use more panels than reserved.
            reserved = len(chosen)
            kept = [s for panel in chosen for s in panel.studs if fitted(s)]
            chosen = sorted(studs.panels(kept)[0], key=lambda p: -p.score)[:reserved]
            sprite_parts = [index[panel.source] for panel in chosen]
            result = bpy.data.objects.new(name + "_LOWPOLY", data)
            output.objects.link(result)
            result.location = origin
            numbers = np.empty(len(data.polygons), dtype=np.int64)
            data.attributes["brickwars_part"].data.foreach_get("value", numbers)
            fallback = [fits[n].color for n in numbers]
            if not settings.bake:
                materials = []
                for n, polygon in zip(numbers, data.polygons):
                    source = fits[n].source.data.materials
                    material = source[0] if source else None
                    if material not in materials:
                        materials.append(material)
                    polygon.material_index = materials.index(material)
                for material in materials:
                    data.materials.append(material)
            # Rays start above the trimmed studs so they are baked as detail.
            extrusion = max(stats["protrusion"] * 1.15, stats["size"] * .003)
            reach = 2 * extrusion + stats["size"] * .01
        else:
            task = rebuild.rebuild(sources, budget, int(settings.detail),
                                   settings.chamfers)
            stage = "Voxelizing the model"
            while True:
                try:
                    fraction = next(task)
                except StopIteration as finished:
                    data, stats = finished.value
                    break
                if fraction >= .2:
                    stage = "Fitting blocks"
                yield .05 + .4 * fraction, stage
            result = bpy.data.objects.new(name + "_LOWPOLY", data)
            output.objects.link(result)
            result.location = origin
            yield .46, "Transferring materials"
            fallback = rebuild.transfer_materials(result, sources)
            # Blocks enclose the source, so rays start just outside each face and
            # travel inward at most as deep as the thickest block.
            extrusion = stats["voxel_size"] * 1.5
            reach = stats["reach"] + extrusion

        if settings.ray_distance:
            extrusion = settings.ray_distance
            reach = 2 * extrusion
        target_obj, moved = result, []
        if settings.generate_uv:
            yield .5, "Unwrapping UV islands"
            if per_part:
                # Identical parts share one region: unwrap a layout with one copy each.
                layout, fallback, moved, shared = parts.atlas_layout(
                    result.data, fits, stats["size"], reach)
                stats.update(shared)
                target_obj = bpy.data.objects.new("BrickWars Atlas Layout", layout)
                temporary.objects.link(target_obj)
                target_obj.location = origin
                uv.generate(context, target_obj, settings.uv_margin, settings.pack, 1.0)
                parts.copy_uvs(target_obj.data, result.data, fits)
            else:
                uv.generate(context, result, settings.uv_margin, settings.pack, 1.0)

        if settings.bake:
            resolution = int(settings.resolution)
            material, shader = atlas.make_material(name + " BrickWars Atlas")
            atlas.assign(result, material)
            if target_obj is not result:
                atlas.assign(target_obj, material)
            # Shared parts are baked alone, from a copy of their source set aside.
            extra = []
            for source, shift in moved:
                copy = bpy.data.objects.new("BrickWars Shared Source", source.data.copy())
                copy.data.transform(Matrix.Translation(Vector(shift)))
                temporary.objects.link(copy)
                copy.location = origin
                extra.append(copy)
            # Cycles bakes much faster from one joined source than from hundreds.
            bake_sources = sources
            if len(sources) + len(extra) > 1:
                everything = sources + extra
                merged = bpy.data.objects.new("BrickWars Bake Source", everything[0].data.copy())
                temporary.objects.link(merged)
                merged.location = origin
                join_into(context, merged, everything[1:], temporary, origin)
                bake_sources = [merged]
            margin = max(1, min(12, int(settings.uv_margin * resolution / 2)))
            # Rounded plastic edges: a fraction of the stud height, but always wide
            # enough to cover a few texels, or the atlas could not show them.
            bevel = 0.0
            if settings.edge_rounding > 0:
                texel = uv.texel_size(target_obj, resolution)
                bevel = settings.edge_rounding * max(.15 * stats.get("protrusion", 0), 1.5 * texel)
                stats["bevel_radius"] = bevel
            nodes = {}
            for index, (kind, normal) in enumerate((('Diffuse', False), ('Normal', True))):
                yield .58 + .18 * index, "Baking " + kind
                target = atlas.image(name + "_" + kind, resolution, normal)
                images.append(target)
                nodes[kind] = atlas.bake(context, target_obj, bake_sources, material, target,
                                         normal, reach, margin, extrusion,
                                         bevel=bevel if normal else 0.0)
                if moved and not normal:
                    placed = bpy.data.objects.new("BrickWars Context Layout",
                                                  parts.in_context(target_obj.data))
                    temporary.objects.link(placed)
                    placed.location = origin
                    behind = atlas.image("BrickWars Context", resolution, False)
                    try:
                        node = atlas.bake(context, placed, bake_sources, material, behind,
                                          False, reach, margin, extrusion)
                        material.node_tree.nodes.remove(node)
                        atlas.fill_missing(target, behind)
                    finally:
                        bpy.data.images.remove(behind)
                    material.node_tree.nodes.active = nodes[kind]
                stats["fallback_pixels"] = stats.get("fallback_pixels", 0) + \
                    atlas.finish_pixels(target, normal, target_obj, fallback)
            occlusion = None
            if settings.lighting > 0:
                yield .84, "Baking studio lighting"
                occlusion = bake_lighting(context, target_obj, bake_sources, material, images[0],
                                          reach, margin, extrusion, settings.lighting,
                                          size=max(max(result.dimensions), 1e-6), bevel=bevel,
                                          tangent=images[1], highlight=settings.edge_highlight,
                                          look=settings.look)
            if settings.specular_alpha:
                yield .9, "Baking specular mask"
                gloss = atlas.image("BrickWars Gloss", resolution, True)
                try:
                    node = atlas.bake(context, target_obj, bake_sources, material, gloss, False,
                                      reach, margin, extrusion, kind='SPEC')
                    material.node_tree.nodes.remove(node)
                    atlas.write_specular(images[0], gloss, occlusion, settings.look)
                finally:
                    bpy.data.images.remove(gloss)
                material.node_tree.nodes.active = nodes['Diffuse']
            atlas.connect(material, shader, nodes['Diffuse'], nodes['Normal'])

        if found:
            stats["studs"] = len(found)
        outputs = list(zip(images, ('Diffuse', 'Normal')))
        repeating = set()
        if chosen:
            yield .92, "Adding stud panels"
            # One repeating texture and material per stud color and layer (wall, cap).
            slots = {}
            for color in sorted({studs.color_key(p.color) for p in chosen}):
                for layer, label in enumerate(studs.LAYERS):
                    kind = f"Studs_{studs.color_name(color)}_{label}"
                    image = studs.texture(name + "_" + kind, color, layer, ratio, settings.look,
                                          settings.lighting, settings.edge_highlight)
                    stud_images.append(image)
                    outputs.append((image, kind))
                    filename = bpy.path.clean_name(name)[:80] + "_" + kind + ".png"
                    stud_material = studs.material(f"{name} BrickWars {kind}", image, filename)
                    stud_materials.append(stud_material)
                    result.data.materials.append(stud_material)
                    slots[color, layer] = len(result.data.materials) - 1
            repeating = set(slots.values())
            studs.add_panels(result.data, chosen, slots, sprite_parts if per_part else None)
            stats["stud_sprites"] = sum(len(p.studs) for p in chosen)
            stats["stud_panels"] = len(chosen)

        yield .94, "Validating result and saving PNGs"
        uv.validate(result, settings.generate_uv or bool(chosen), repeating)
        after = mesh.triangles(result.data)
        if outputs:
            root = Path(bpy.path.abspath(settings.directory))
            if settings.directory.startswith("//") and not bpy.data.filepath:
                root = Path(tempfile.gettempdir()) / "BrickWars"
            root.mkdir(parents=True, exist_ok=True)
            directory = root / (bpy.path.clean_name(name)[:48] + "_" + uuid4().hex[:12])
            directory.mkdir()
            for target, kind in outputs:
                path = directory / (bpy.path.clean_name(name)[:80] + "_" + kind + ".png")
                target.filepath_raw = str(path)
                target.save()
                if path.stat().st_size < 64:
                    raise ValueError("PNG output is incomplete.")
        if per_part:
            joined = result
            pieces = parts.split(joined, fits, output, [source_objects[i] for i in keep])
            mesh.remove_object(joined)
            result = None
            if not pieces:
                raise ValueError("Every part was hidden.")
            result = pieces[0]
            for piece in pieces:
                piece["brickwars_part"] = True
            stats["visible_parts"] = len(pieces)
        result["brickwars_method"] = settings.method
        result["brickwars_triangles_before"] = before
        result["brickwars_triangles_after"] = after
        result["brickwars_target_met"] = guided or after <= settings.target
        for key in ("blocks", "parts", "visible_parts", "hidden_faces", "mean_iou",
                    "overfill", "fallback_pixels", "part_types", "shared_parts", "bevel_radius",
                    "texture_groups", "studs", "stud_sprites", "stud_panels"):
            if key in stats:
                result["brickwars_" + key] = stats[key]
        if directory:
            result["brickwars_texture_directory"] = str(directory)
        complete = True
        return result
    finally:
        if temporary:
            for obj in list(temporary.objects):
                mesh.remove_object(obj)
            bpy.data.collections.remove(temporary)
        if complete:
            mesh.activate(context, result, pieces[1:])
        else:
            if result:
                mesh.remove_object(result)
            if output:
                bpy.data.collections.remove(output)
            if material:
                bpy.data.materials.remove(material)
            for stud_material in stud_materials:
                bpy.data.materials.remove(stud_material)
            for image in stud_images:
                bpy.data.images.remove(image)
            for image in images:
                bpy.data.images.remove(image)
            if directory:
                for path in directory.glob("*.png"):
                    path.unlink()
                directory.rmdir()
            for obj in selection:
                obj.select_set(True)
            context.view_layer.objects.active = active
