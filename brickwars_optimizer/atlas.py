"""Cycles baking from the dense source onto the rebuilt low-poly mesh."""
import bpy
import numpy as np

from .mesh import activate


def make_material(name):
    material = bpy.data.materials.new(name)
    nodes = material.node_tree.nodes
    shader = next(n for n in nodes if n.type == 'BSDF_PRINCIPLED')
    shader.inputs['Roughness'].default_value = .6
    return material, shader


def texture_node(material, image):
    nodes = material.node_tree.nodes
    node = nodes.new('ShaderNodeTexImage')
    node.image = image
    node.label = image.name
    nodes.active = node
    return node


def assign(obj, material):
    obj.data.materials.clear()
    obj.data.materials.append(material)
    for face in obj.data.polygons:
        face.material_index = 0


def image(name, size, normal):
    result = bpy.data.images.new(name, width=size, height=size, alpha=True)
    result.colorspace_settings.name = 'Non-Color' if normal else 'sRGB'
    result.file_format = 'PNG'
    return result


PLASTIC_GLOSS = .7


def specular_value(material):
    """Gloss of the part for the specular mask: ABS plastic, more for metal and clear parts."""
    shader = None
    if material and material.node_tree:
        shader = next((n for n in material.node_tree.nodes if n.type == 'BSDF_PRINCIPLED'), None)
    if shader is None:
        return PLASTIC_GLOSS

    def value(name, default):
        socket = shader.inputs.get(name)
        return default if socket is None or socket.is_linked else socket.default_value

    if (value('Metallic', 0) > .5 or value('Alpha', 1) < .9
            or value('Transmission Weight', 0) > .5):
        return 1.0
    # Most imports leave roughness at zero; only clearly rough materials lose gloss.
    return PLASTIC_GLOSS * (1 - .7 * min(max(value('Roughness', 0) - .3, 0) / .5, 1))


def bevel_material(radius):
    """Neutral surface with rounded edges, used for normal bakes of the source."""
    material = bpy.data.materials.new("BrickWars Rounded Edges")
    nodes = material.node_tree.nodes
    shader = next(n for n in nodes if n.type == 'BSDF_PRINCIPLED')
    bevel = nodes.new('ShaderNodeBevel')
    bevel.samples = 16
    bevel.inputs['Radius'].default_value = radius
    material.node_tree.links.new(bevel.outputs['Normal'], shader.inputs['Normal'])
    return material


def emission_copy(material, value=None):
    copy = material.copy() if material else bpy.data.materials.new("BrickWars Source Color")
    tree = copy.node_tree
    nodes = tree.nodes
    output = next((n for n in nodes if n.type == 'OUTPUT_MATERIAL' and n.is_active_output), None)
    if output is None:
        output = nodes.new('ShaderNodeOutputMaterial')
    socket = output.inputs['Surface']
    shader = socket.links[0].from_node if socket.is_linked else None
    emit = nodes.new('ShaderNodeEmission')
    if value is not None:
        emit.inputs['Color'].default_value = (value, value, value, 1)
    elif shader and shader.type in {'BSDF_PRINCIPLED', 'BSDF_DIFFUSE', 'EMISSION'}:
        source = shader.inputs['Base Color' if shader.type == 'BSDF_PRINCIPLED' else 'Color']
        if source.is_linked:
            tree.links.new(source.links[0].from_socket, emit.inputs['Color'])
        else:
            emit.inputs['Color'].default_value = source.default_value
    else:
        emit.inputs['Color'].default_value = material.diffuse_color if material else (.6,.6,.6,1)
    tree.links.new(emit.outputs[0], socket)
    return copy


def bake(context, obj, sources, material, target, normal, distance, margin, extrusion=None,
         kind=None, samples=1, bevel=0.0):
    """kind: None (emission color or tangent normal), 'AO', 'OBJECT' (object-space normal)
    or 'SPEC' (gloss of each source material). bevel rounds source edges in normal bakes."""
    scene = context.scene
    engine, device = scene.render.engine, scene.cycles.device
    previous_samples = scene.cycles.samples
    node = texture_node(material, target)
    copies = {}
    original_slots = []
    added_slots = []
    try:
        if not normal:
            def copy_of(original):
                if original not in copies:
                    copies[original] = emission_copy(
                        original, specular_value(original) if kind == 'SPEC' else None)
                return copies[original]
            for source in sources:
                if not source.data.materials:
                    source.data.materials.append(copy_of(None))
                    added_slots.append(source)
                    continue
                for slot in source.material_slots:
                    original = slot.material
                    original_slots.append((slot, original))
                    slot.material = copy_of(original)
        elif bevel > 0 and kind != 'AO':
            copies['bevel'] = rounded = bevel_material(bevel)
            samples = max(samples, 8)
            for source in sources:
                if not source.data.materials:
                    source.data.materials.append(rounded)
                    added_slots.append(source)
                    continue
                for slot in source.material_slots:
                    original_slots.append((slot, slot.material))
                    slot.material = rounded
        else:
            for source in sources:
                if not source.data.materials:
                    if None not in copies:
                        copies[None] = emission_copy(None)
                    source.data.materials.append(copies[None])
                    added_slots.append(source)
                for slot in source.material_slots:
                    if slot.material is None:
                        if None not in copies:
                            copies[None] = emission_copy(None)
                        original_slots.append((slot, None))
                        slot.material = copies[None]
        scene.render.engine = 'CYCLES'
        scene.cycles.device = 'CPU'
        scene.cycles.samples = samples
        activate(context, obj, sources)
        status = bpy.ops.object.bake(
            type='AO' if kind == 'AO' else ('NORMAL' if normal else 'EMIT'),
            use_selected_to_active=True, use_clear=True,
            normal_space='OBJECT' if kind == 'OBJECT' else 'TANGENT', margin=margin, margin_type='EXTEND',
            max_ray_distance=distance*2 if extrusion is None else distance,
            cage_extrusion=distance if extrusion is None else extrusion, save_mode='INTERNAL',
            target='IMAGE_TEXTURES', uv_layer=obj.data.uv_layers.active.name,
        )
        if 'FINISHED' not in status:
            raise ValueError("Cycles could not finish baking.")
    finally:
        scene.cycles.samples = previous_samples
        scene.cycles.device = device
        scene.render.engine = engine
        for slot, original in original_slots:
            slot.material = original
        for source in added_slots:
            source.data.materials.clear()
        for copy in copies.values():
            bpy.data.materials.remove(copy)
    return node


def srgb_color(color):
    linear = np.clip(color, 0, 1)
    return np.where(linear <= .0031308, 12.92*linear, 1.055*linear**(1/2.4)-.055)


def fill_unprojected(pixels, covered, obj, fallback, padding=2):
    """Keep the rebuilt material wherever projection cannot find the source."""
    size = pixels.shape[0]
    layer = obj.data.uv_layers.active.uv
    count = 0
    for polygon, color in zip(obj.data.polygons, fallback):
        coordinates = np.asarray([tuple(layer[i].vector) for i in polygon.loop_indices])*size
        origin = coordinates[0]
        basis = (coordinates[1:] - origin).T
        if abs(np.linalg.det(basis)) < 1e-8:
            continue
        inverse = np.linalg.inv(basis)
        low = np.maximum(np.floor(coordinates.min(axis=0)).astype(int)-padding, 0)
        high = np.minimum(np.ceil(coordinates.max(axis=0)).astype(int)+padding, size)
        x, y = np.meshgrid(np.arange(low[0], high[0])+.5, np.arange(low[1], high[1])+.5)
        bary = (np.stack((x, y), axis=-1) - origin) @ inverse.T
        thresholds = padding*np.linalg.norm(inverse, axis=1)
        inside = np.all(bary >= -thresholds, axis=-1)
        inside &= bary.sum(axis=-1) <= 1+padding*np.linalg.norm(inverse.sum(axis=0))
        patch = np.s_[low[1]:high[1], low[0]:high[0]]
        mask = covered[patch]
        missing = inside & ~mask
        pixels[patch][missing, :3] = srgb_color(color)
        mask[missing] = True
        count += int(np.count_nonzero(missing))
    return count


def fill_missing(target, behind):
    """Take texels the first bake missed from a second bake of the same UV layout."""
    pixels, other = pixels_of(target), pixels_of(behind)
    missing = (pixels[:, :, 3] <= .5) & (other[:, :, 3] > .5)
    pixels[missing] = other[missing]
    target.pixels.foreach_set(pixels.ravel())
    target.update()
    return int(np.count_nonzero(missing))


def finish_pixels(target, normal, obj, fallback=None):
    size = target.size[0]
    pixels = np.empty((size, size, 4), dtype=np.float32)
    target.pixels.foreach_get(pixels.ravel())
    opaque = pixels[:, :, 3] > .5
    if not opaque.any():
        raise ValueError("The bake was empty. Check the geometry and projection distance.")
    filled = 0
    if normal:
        pixels[~opaque] = (.5, .5, 1, 1)
    else:
        if fallback is not None:
            filled = fill_unprojected(pixels, opaque, obj, fallback)
        pixels[:, :, 3] = 1
    target.pixels.foreach_set(pixels.ravel())
    target.update()
    return filled


def connect(material, shader, diffuse, normal):
    links = material.node_tree.links
    links.new(diffuse.outputs['Color'], shader.inputs['Base Color'])
    decoder = material.node_tree.nodes.new('ShaderNodeNormalMap')
    decoder.space = 'TANGENT'
    links.new(normal.outputs['Color'], decoder.inputs['Color'])
    links.new(decoder.outputs['Normal'], shader.inputs['Normal'])


def to_linear(srgb):
    srgb = np.clip(srgb, 0, 1)
    return np.where(srgb <= .04045, srgb / 12.92, ((srgb + .055) / 1.055) ** 2.4)


def pixels_of(image):
    size = image.size[0]
    values = np.empty((size, size, 4), dtype=np.float32)
    image.pixels.foreach_get(values.ravel())
    return values


def smooth(values, mask, radius):
    """Box blur that only averages covered texels, so islands do not bleed."""
    weight = mask.astype(np.float32)
    total = values * weight
    for axis in (0, 1):
        sums, counts = np.zeros_like(total), np.zeros_like(weight)
        for shift in range(-radius, radius + 1):
            sums += np.roll(total, shift, axis=axis)
            counts += np.roll(weight, shift, axis=axis)
        total, weight = sums, counts
    return np.where(mask, total / np.maximum(weight, 1e-6), values)


# Look presets for the baked diffuse atlas.
#   shade:     base, key and sky weights of the light from above (top face = 1)
#   ao:        floor and exponent of the ambient occlusion
#   cavity:    dark lines in narrow crevices, so parts read clearly apart
#   saturation, exposure: color grading in linear space
#   edge:      gain and high-pass radius of the edge highlights
#   gloss:     specular mask for plain plastic
LOOKS = {
    'STUDIO': dict(shade=(.45, .35, .2), ao=(.3, .8), cavity=0.0, saturation=1.0,
                   exposure=1.0, edge=(.45, 4), gloss=.7),
    # Clean, saturated toy look of console LEGO games: bright flat light (the engine
    # adds its own), light occlusion with crisp crevice lines, strong narrow highlights.
    'GAME': dict(shade=(.64, .22, .14), ao=(.6, 1.2), cavity=.55, saturation=1.25,
                 exposure=1.06, edge=(.65, 2), gloss=.85),
}


def apply_lighting(diffuse, occlusion, normals, strength, tangent=None, highlight=0.0,
                   look='STUDIO'):
    """Bake the look: ambient occlusion, a soft key light from above and bright edges.

    tangent is the baked tangent-space normal map. Where it bends away from the flat
    face (rounded edges, stud rims) the plastic catches the light, as in a render.
    Returns the smoothed occlusion for the specular mask.
    """
    preset = LOOKS[look]
    color = pixels_of(diffuse)
    ao = pixels_of(occlusion)
    n = pixels_of(normals)
    covered = (ao[:, :, 3] > .5) & (n[:, :, 3] > .5)
    direction = n[:, :, :3] * 2 - 1
    direction /= np.maximum(np.linalg.norm(direction, axis=-1, keepdims=True), 1e-6)
    # Soft light from straight above: the same for any turn about the vertical axis,
    # so identical parts sharing one texture region look right in every heading.
    up = direction[:, :, 2]
    key = np.clip(.2 + .8 * up, 0, 1)
    sky = .5 + .5 * up
    # A face pointing straight up keeps its material color.
    base, key_weight, sky_weight = preset['shade']
    shade = base + key_weight * key + sky_weight * sky
    occlusion_value = np.clip(smooth(ao[:, :, 0], covered, 2), 0, 1)
    floor, power = preset['ao']
    occ = floor + (1 - floor) * occlusion_value ** power
    if preset['cavity'] > 0:
        # Narrow crevices are darker than their surroundings: a thin line, not a shadow.
        crevice = np.clip((smooth(occlusion_value, covered, 5) - occlusion_value) * 6, 0, 1)
        occ = occ * (1 - preset['cavity'] * crevice)
    factor = np.where(covered, shade * occ, 1.0)
    factor = 1 + strength * (factor - 1)
    linear = to_linear(color[:, :, :3])
    if preset['saturation'] != 1 or preset['exposure'] != 1:
        grade = 1 + strength * (preset['saturation'] - 1)
        luminance = (linear @ np.array([.2126, .7152, .0722]))[:, :, None]
        graded = np.clip((luminance + (linear - luminance) * grade)
                         * (1 + strength * (preset['exposure'] - 1)), 0, 1)
        linear = np.where(covered[:, :, None], graded, linear)
    linear = linear * factor[:, :, None]
    if tangent is not None and highlight > 0:
        gain_weight, radius = preset['edge']
        bent = pixels_of(tangent)[:, :, :3] * 2 - 1
        # High-pass: only narrow bends count, not gently curved surfaces.
        local = np.stack([smooth(bent[:, :, c], covered, radius) for c in range(3)], axis=-1)
        sharp = np.linalg.norm(bent - local, axis=-1)
        edge = np.clip(sharp * 4, 0, 1) * (.35 + .65 * np.clip(up, 0, 1))
        edge = np.where(covered, edge * occlusion_value, 0)
        # Screen blend: highlights lighten dark plastic too, but never clip to white.
        gain = (highlight * gain_weight * edge)[:, :, None]
        linear = linear + (1 - linear) * gain
    color[:, :, :3] = srgb_color(linear)
    diffuse.pixels.foreach_set(color.ravel())
    diffuse.update()
    return np.where(covered, occlusion_value, 1.0)


def write_specular(diffuse, gloss, occlusion=None, look='STUDIO'):
    """Store the gloss in the diffuse alpha, as SWBF2 reads it for specular materials."""
    color = pixels_of(diffuse)
    values = pixels_of(gloss)
    hit = values[:, :, 3] > .5
    # Material gloss is measured against plain plastic, then scaled to the look.
    mask = np.where(hit, values[:, :, 0], PLASTIC_GLOSS) * (LOOKS[look]['gloss'] / PLASTIC_GLOSS)
    if occlusion is not None:
        # Crevices and holes do not reflect the light.
        mask = mask * (.25 + .75 * occlusion)
    color[:, :, 3] = np.clip(mask, 0, 1)
    diffuse.pixels.foreach_set(color.ravel())
    diffuse.update()
