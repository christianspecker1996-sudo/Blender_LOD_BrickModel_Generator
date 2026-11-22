"""Find LEGO studs, take them out of the bake source and fake them with layered sprites.

A stud is a flat circular cap on a short cylinder (Studio and LDD often add a bevel
ring and leave the mesh open at seams, so it is found by geometry, not topology):
the cap boundary is a circle, the surface around the stud lies one stud height
lower, and the faces in between form an outward-facing wall.

Removed studs leave a flat disc, so the atlas shows a plain surface. As in older
console LEGO games, each visible stud can then be drawn by two stacked
alpha-tested quads with their own small texture and material: a dark, slightly
larger disc just above the surface (wall and contact shadow) and the bright cap
at full stud height. Seen at an angle, the offset between them reads as depth.
"""
from collections import Counter
from dataclasses import dataclass
from math import atan2, cos, pi, sin, sqrt

import bmesh
import bpy
import numpy as np
from mathutils import Vector
from mathutils.bvhtree import BVHTree

from . import atlas
from .rebuild import material_color

TRIS_PER_STUD = 4
TILE = 128
DISC_SIDES = 16


@dataclass
class Stud:
    base: Vector        # center of the stud on the surface it stands on
    axis: Vector        # outward direction of the cap
    radius: float       # outer wall radius
    height: float
    color: tuple        # linear RGB of the stud material
    faces: list         # face indices in the source mesh
    material: int
    source: int = 0     # index of the source object
    score: int = 0      # rays that escape from the cap (0 = covered)

    @property
    def top(self):
        return self.base + self.axis * self.height


# --------------------------------------------------------------------------- detection

def boundary(faces):
    return [edge for edge, count in Counter(
        edge for face in faces for edge in face.edges).items() if count == 1]


def flood(seed, allowed):
    found = {seed}
    stack = [seed]
    while stack:
        for edge in stack.pop().edges:
            for face in edge.link_faces:
                if face not in found and allowed(face):
                    found.add(face)
                    stack.append(face)
    return found


def ring(edges, axis):
    """Center and radius if the edges form one closed circle perpendicular to axis."""
    degrees = Counter(v for edge in edges for v in edge.verts)
    if len(degrees) < 8 or any(degree != 2 for degree in degrees.values()):
        return None
    center = sum((v.co for v in degrees), Vector()) / len(degrees)
    lengths = [(v.co - center).length for v in degrees]
    radius = sum(lengths) / len(lengths)
    if radius <= 1e-10 or max(abs(length - radius) for length in lengths) > radius * .065:
        return None
    if max(abs((v.co - center).dot(axis)) for v in degrees) > radius * .015:
        return None
    tangent = (next(iter(degrees)).co - center).normalized()
    bitangent = axis.cross(tangent)
    angles = sorted(atan2((v.co-center).dot(bitangent), (v.co-center).dot(tangent))
                    for v in degrees)
    if max(b-a for a, b in zip(angles, angles[1:] + [angles[0]+2*pi])) > 1.05:
        return None
    return center, radius


def frame(axis):
    tangent = axis.orthogonal().normalized()
    return tangent, axis.cross(tangent)


def base_height(tree, top, axis, radius):
    """Depth of the surface around the stud, probed just outside its wall."""
    tangent, side = frame(axis)
    lift = .05 * radius
    depths = []
    for direction in (tangent, -tangent, side, -side):
        probe = top + direction * (1.45 * radius) + axis * lift
        hit, normal, _, distance = tree.ray_cast(probe, -axis, 1.6 * radius)
        if hit is not None and normal.dot(axis) > .95:
            depths.append(distance - lift)
    if len(depths) < 3:
        return None
    depths.sort()
    middle = depths[len(depths) // 2]
    if sum(abs(d - middle) < .05 * radius for d in depths) < 3:
        return None
    return middle


def stud_faces(bm, tree, cap, top, axis, cap_radius, height):
    """Faces between cap and base surface, or None if they do not form a stud."""
    reach = sqrt((height / 2 + .05 * cap_radius) ** 2 + (1.4 * cap_radius) ** 2)
    found = set(cap)
    wall_area = 0.0
    outer = cap_radius
    low, high = -height - .03 * cap_radius, .03 * cap_radius
    for _, _, index, _ in tree.find_nearest_range(top - axis * (height / 2), reach):
        face = bm.faces[index]
        if face in found:
            continue
        radial_max, z_sum, inside = 0.0, 0.0, True
        for vert in face.verts:
            delta = vert.co - top
            z = delta.dot(axis)
            radial = (delta - z * axis).length
            if not low <= z <= high or radial > 1.3 * cap_radius:
                inside = False
                break
            radial_max = max(radial_max, radial)
            z_sum += z
        if not inside:
            continue
        z_mean = z_sum / len(face.verts)
        facing = face.normal.dot(axis)
        if facing > .98 and z_mean < -.5 * height:
            continue                    # the surface the stud stands on
        if facing < -.98 and z_mean < -.9 * height:
            found.add(face)             # bottom of a stud modelled as its own shell
            continue
        if facing < -.3:
            return None                 # overhang: not a plain stud
        if abs(facing) < .3:
            middle = face.calc_center_median() - top
            outward = middle - middle.dot(axis) * axis
            if outward.length < 1e-12 or face.normal.dot(outward.normalized()) < .3:
                return None             # inner wall: hollow stud or tube
            wall_area += face.calc_area()
        outer = max(outer, radial_max)
        found.add(face)
    if wall_area < .5 * 2 * pi * cap_radius * height:
        return None
    return found, outer


def detect(data):
    """All studs of a mesh (in its own space). The mesh is not changed."""
    bm = bmesh.new()
    try:
        bm.from_mesh(data)
        bm.normal_update()
        bm.faces.ensure_lookup_table()
        tree = BVHTree.FromBMesh(bm)
        seen, occupied, result = set(), set(), []
        for face in bm.faces:
            if face in seen or face.calc_area() < 1e-20:
                continue
            axis = face.normal.copy()
            point = face.verts[0].co.copy()
            epsilon = max(face.calc_area() ** .5 * .0005, 1e-9)
            cap = flood(face, lambda f: f.normal.dot(axis) > .9999 and
                        abs((f.verts[0].co - point).dot(axis)) < epsilon)
            seen.update(cap)
            if cap & occupied:
                continue
            circle = ring(boundary(cap), axis)
            if not circle:
                continue
            top, cap_radius = circle
            height = base_height(tree, top, axis, cap_radius)
            if height is None or not .35 < height / cap_radius < 1.2:
                continue
            region = stud_faces(bm, tree, cap, top, axis, cap_radius, height)
            if not region or region[0] & occupied:
                continue
            faces, outer = region
            occupied.update(faces)
            index = face.material_index
            material = data.materials[index] if index < len(data.materials) else None
            result.append(Stud(top - axis * height, axis, outer, height,
                               tuple(material_color(material)),
                               sorted(f.index for f in faces), index))
        return result
    finally:
        bm.free()


def remove(data, studs):
    """Delete the studs and close each footprint with a flat disc of the stud color."""
    if not studs:
        return
    bm = bmesh.new()
    try:
        bm.from_mesh(data)
        bm.faces.ensure_lookup_table()
        doomed = {bm.faces[i] for stud in studs for i in stud.faces}
        bmesh.ops.delete(bm, geom=list(doomed), context='FACES_ONLY')
        for edge in [e for e in bm.edges if not e.link_faces]:
            bm.edges.remove(edge)
        for vert in [v for v in bm.verts if not v.link_edges]:
            bm.verts.remove(vert)
        for stud in studs:
            tangent, side = frame(stud.axis)
            radius = stud.radius * 1.02
            verts = [bm.verts.new(stud.base + radius * (cos(a) * tangent + sin(a) * side))
                     for a in (2 * pi * k / DISC_SIDES for k in range(DISC_SIDES))]
            disc = bm.faces.new(verts)
            disc.material_index = stud.material
            disc.smooth = False
            bmesh.ops.triangulate(bm, faces=[disc])
        bm.to_mesh(data)
        data.update()
    finally:
        bm.free()


# --------------------------------------------------------------------------- visibility

def hemisphere(normal):
    tangent, bitangent = frame(normal)
    directions = [normal]
    for step in range(8):
        angle = step * pi / 4
        side = tangent * cos(angle) + bitangent * sin(angle)
        directions.append((normal * .65 + side * .76).normalized())
    return directions


def extent(sources):
    """Longest side of the box around all sources."""
    low, high = np.full(3, np.inf), np.full(3, -np.inf)
    for source in sources:
        co = np.empty(len(source.data.vertices) * 3)
        source.data.vertices.foreach_get("co", co)
        if len(co):
            co = co.reshape(-1, 3)
            low, high = np.minimum(low, co.min(axis=0)), np.maximum(high, co.max(axis=0))
    return float(max((high - low).max(), 1e-6)) if np.isfinite(low).all() else 1.0


def score(studs, sources, size):
    """Count for each stud the rays from its cap that reach open space."""
    verts, polys = [], []
    for source in sources:
        data = source.data
        data.calc_loop_triangles()
        co = np.empty(len(data.vertices) * 3)
        data.vertices.foreach_get("co", co)
        base = len(verts)
        verts.extend(map(tuple, co.reshape(-1, 3)))
        polys.extend([base + v for v in tri.vertices] for tri in data.loop_triangles)
    if not polys:
        return
    tree = BVHTree.FromPolygons(verts, polys, all_triangles=True)
    for stud in studs:
        start = stud.top + stud.axis * max(.1 * stud.height, 1e-5 * size)
        stud.score = sum(tree.ray_cast(start, direction, size * 4)[0] is None
                         for direction in hemisphere(stud.axis))


def on_shape(stud, normals, offsets):
    """True if the stud stands on the surface of a convex low-poly shape.

    The shape (half-spaces normals @ p <= offsets) must leave the stud out, or the
    stud is part of the shape (a pin end, a bar) and must stay in the bake source;
    its foot must touch the shape, or it would float above it."""
    if normals is None:
        return False
    normals = np.asarray(normals)
    offsets = np.asarray(offsets)
    tolerance = .3 * stud.height
    top = normals @ np.asarray(stud.base + stud.axis * (.7 * stud.height)) - offsets
    base = normals @ np.asarray(stud.base) - offsets
    return bool(top.max() > tolerance and base.max() < tolerance)




# --------------------------------------------------------------------------- panels

@dataclass
class Panel:
    """A gap-free rectangle of studs on one lattice, drawn as two stacked quads.

    The stud texture repeats once per lattice cell, so a whole stud field costs as
    much as a single stud: a copy of the surface just above it shows the walls and
    a copy at stud height shows the caps."""
    studs: list
    corner: Vector      # corner at the foot of the studs
    u: Vector           # full edge along the columns
    v: Vector           # full edge along the rows
    cells: tuple        # (columns, rows)

    axis = property(lambda self: self.studs[0].axis)
    color = property(lambda self: self.studs[0].color)
    source = property(lambda self: self.studs[0].source)
    height = property(lambda self: sum(s.height for s in self.studs) / len(self.studs))
    score = property(lambda self: sum(s.score for s in self.studs))


# Stud spacing over stud radius when no neighbour shows it (8 mm over 2.4-2.55 mm).
PITCH_RATIO = 3.2


def lattice(group):
    """Pitch and in-plane unit vectors of coplanar studs, or None."""
    if len(group) < 2:
        return None
    axis = group[0].axis
    steps = [min((b.base - a.base for b in group if b is not a), key=lambda d: d.length)
             for a in group]
    lengths = sorted(d.length for d in steps)
    pitch = lengths[len(lengths) // 2]
    radius = sum(s.radius for s in group) / len(group)
    if not 2.6 * radius < pitch < 4.2 * radius:
        return None
    step = next(d for d in steps if abs(d.length - pitch) < .05 * pitch)
    u = step - step.dot(axis) * axis
    if u.length < .5 * pitch:
        return None
    u.normalize()
    return pitch, u, axis.cross(u)


def rectangles(cells):
    """Split occupied lattice cells into gap-free rectangles, longest rows first."""
    left = set(cells)
    while left:
        j0, i0 = min((j, i) for i, j in left)
        i1 = i0
        while (i1 + 1, j0) in left:
            i1 += 1
        j1 = j0
        while all((i, j1 + 1) in left for i in range(i0, i1 + 1)):
            j1 += 1
        for i in range(i0, i1 + 1):
            for j in range(j0, j1 + 1):
                left.discard((i, j))
        yield i0, j0, i1, j1


def panels(studs):
    """Group studs of one part, plane, height and color into lattice rectangles.
    Returns (panels, pitch over radius)."""
    groups = {}
    for stud in studs:
        unit = .1 * stud.radius
        key = (stud.source, tuple(round(c, 2) for c in stud.axis),
               round(stud.base.dot(stud.axis) / unit), round(stud.height / unit),
               color_key(stud.color))
        groups.setdefault(key, []).append(stud)
    result, singles, ratios = [], [], []
    for group in groups.values():
        grid = lattice(group)
        if grid is None:
            singles.extend(group)
            continue
        pitch, u, v = grid
        ratios.append(pitch / (sum(s.radius for s in group) / len(group)))
        origin = group[0].base
        cells = {}
        for stud in group:
            delta = stud.base - origin
            i, j = delta.dot(u) / pitch, delta.dot(v) / pitch
            key = (round(i), round(j))
            if abs(i - key[0]) > .12 or abs(j - key[1]) > .12 or key in cells:
                singles.append(stud)
                continue
            cells[key] = stud
        for i0, j0, i1, j1 in rectangles(cells):
            members = [cells[i, j] for i in range(i0, i1 + 1) for j in range(j0, j1 + 1)]
            corner = origin + ((i0 - .5) * u + (j0 - .5) * v) * pitch
            columns, rows = i1 - i0 + 1, j1 - j0 + 1
            result.append(Panel(members, corner, u * pitch * columns, v * pitch * rows,
                                (columns, rows)))
    ratio = sorted(ratios)[len(ratios) // 2] if ratios else PITCH_RATIO
    for stud in singles:
        pitch = stud.radius * ratio
        tangent, side = frame(stud.axis)
        corner = stud.base - (tangent + side) * (pitch / 2)
        result.append(Panel([stud], corner, tangent * pitch, side * pitch, (1, 1)))
    return result, ratio


def choose(found, budget):
    """Panels of visible studs within the triangle budget, most exposed first."""
    candidates, ratio = panels([s for s in found if s.score > 0])
    candidates.sort(key=lambda p: -p.score)
    return candidates[:max(0, int(budget // TRIS_PER_STUD))], ratio


# --------------------------------------------------------------------------- texture

def color_key(color):
    return tuple(round(c, 3) for c in color)


def color_name(color):
    return "".join("%02X" % int(round(c * 255))
                   for c in atlas.srgb_color(np.array(color[:3], dtype=np.float32)))


def graded(color, look, strength):
    """Material color graded like the baked atlas (saturation and exposure of the look)."""
    preset = atlas.LOOKS[look]
    linear = np.clip(np.array(color[:3], dtype=np.float32), 0, 1)
    luminance = float(linear @ np.array([.2126, .7152, .0722]))
    grade = 1 + strength * (preset['saturation'] - 1)
    exposure = 1 + strength * (preset['exposure'] - 1)
    return np.clip((luminance + (linear - luminance) * grade) * exposure, 0, 1)


LAYERS = ("Wall", "Cap")
WALL_SCALE = 1.1


def tile_pixels(color, layer, look, strength, highlight, disc):
    """One lattice cell as RGBA: layer 0 the shaded wall disc, layer 1 the lit cap.
    disc is the disc radius over half the cell width."""
    preset = atlas.LOOKS[look]
    coords = (np.arange(TILE) + .5) / TILE * 2 - 1
    x, y = np.meshgrid(coords, coords)
    r = np.sqrt(x * x + y * y) / disc          # 1 at the disc edge
    base = graded(color, look, strength)
    if layer == 0:
        # The wall seen from above: in shade, darkest at the contact line.
        wall = .5 if look == 'STUDIO' else .62
        shade = 1 + strength * (wall * (1 - .18 * np.clip(r, 0, 1)) - 1)
        if preset['cavity'] > 0:
            shade = shade * (1 - strength * preset['cavity'] * .5 * (r > .9))
        linear = base[None, None, :] * shade[:, :, None]
    else:
        # The cap: flat plastic lit from above with a bright rounded rim.
        shade = 1 - strength * .06 * np.clip(r, 0, 1) ** 2
        linear = base[None, None, :] * shade[:, :, None]
        gain_weight, _ = preset['edge']
        rim = np.clip((r - .74) / .2, 0, 1) * (r <= 1)
        gain = (highlight * gain_weight * 1.2 * rim)[:, :, None]
        linear = linear + (1 - linear) * gain
    pixels = np.empty((TILE, TILE, 4), dtype=np.float32)
    pixels[:, :, :3] = atlas.srgb_color(np.clip(linear, 0, 1))
    pixels[:, :, 3] = (r <= 1).astype(np.float32)
    return pixels


def texture(name, color, layer, ratio, look, strength, highlight):
    """Repeating texture with one stud per image, for one color and layer."""
    disc = 2 / ratio * (WALL_SCALE if layer == 0 else 1.0)
    pixels = tile_pixels(color, layer, look, strength, highlight, min(disc, .98))
    image = bpy.data.images.new(name, width=TILE, height=TILE, alpha=True)
    image.colorspace_settings.name = 'sRGB'
    image.alpha_mode = 'STRAIGHT'
    image.file_format = 'PNG'
    image.pixels.foreach_set(pixels.ravel())
    image.update()
    return image


def material(name, image, filename):
    """Alpha-tested stud material with a repeating texture (SWBF2: hardedged)."""
    result = bpy.data.materials.new(name)
    tree = result.node_tree
    shader = next(n for n in tree.nodes if n.type == 'BSDF_PRINCIPLED')
    shader.inputs['Roughness'].default_value = .6
    node = tree.nodes.new('ShaderNodeTexImage')
    node.image = image
    node.interpolation = 'Linear'
    node.extension = 'REPEAT'
    node.location = (-320, 200)
    tree.nodes.active = node
    tree.links.new(node.outputs['Color'], shader.inputs['Base Color'])
    tree.links.new(node.outputs['Alpha'], shader.inputs['Alpha'])
    for attribute, value in (("surface_render_method", 'DITHERED'), ("blend_method", 'CLIP'),
                             ("use_backface_culling", True)):
        try:
            setattr(result, attribute, value)
        except (AttributeError, TypeError, ValueError):
            pass
    result["brickwars_studs"] = True
    result["swbf_hardedged"] = True
    # SWBF msh exporter (if installed): hardedged transparency with this texture.
    props = getattr(result, "swbf_msh_mat", None)
    if props is not None:
        for attribute, value in (("hardedged_transparency", True),
                                 ("diffuse_map", filename)):
            try:
                setattr(props, attribute, value)
            except (AttributeError, TypeError, ValueError):
                pass
    return result


# --------------------------------------------------------------------------- sprites

def add_panels(data, chosen, slots, parts=None):
    """Append two stacked quads per panel. UVs count lattice cells, so the texture
    repeats once per stud. slots maps (color key, layer) to a material index; parts
    gives each panel's part number."""
    bm = bmesh.new()
    try:
        bm.from_mesh(data)
        uv = bm.loops.layers.uv.active or bm.loops.layers.uv.new("BrickWars UV")
        part_layer = bm.faces.layers.int.get("brickwars_part")
        corners = [(0, 0), (1, 0), (1, 1), (0, 1)]
        for number, panel in enumerate(chosen):
            columns, rows = panel.cells
            height = panel.height
            for layer, lift in enumerate((max(.04 * height, 1e-6), height)):
                start = panel.corner + panel.axis * lift
                verts = [bm.verts.new(start + a * panel.u + b * panel.v) for a, b in corners]
                index = slots[color_key(panel.color), layer]
                for indices in ((0, 1, 2), (0, 2, 3)):
                    face = bm.faces.new([verts[i] for i in indices])
                    face.material_index = index
                    face.smooth = False
                    if part_layer is not None and parts is not None:
                        face[part_layer] = parts[number]
                    for loop, i in zip(face.loops, indices):
                        a, b = corners[i]
                        loop[uv].uv = (a * columns, b * rows)
        bm.to_mesh(data)
        data.update()
    finally:
        bm.free()
