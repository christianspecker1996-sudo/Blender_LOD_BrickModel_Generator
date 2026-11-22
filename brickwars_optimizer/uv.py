"""Create and validate the single game UV map."""
import math

import bpy

from .mesh import activate


def generate(context, obj, margin, pack, surface_fraction):
    mesh = obj.data
    for layer in list(mesh.uv_layers):
        mesh.uv_layers.remove(layer)
    mesh.uv_layers.new(name="BrickWars UV")
    activate(context, obj)
    bpy.ops.object.mode_set(mode='EDIT')
    try:
        bpy.ops.mesh.select_all(action='SELECT')
        bpy.ops.uv.smart_project(angle_limit=math.radians(66), island_margin=margin,
                                 margin_method='SCALED')
        if pack:
            bpy.ops.uv.pack_islands(margin=margin, margin_method='ADD',
                                    shape_method='CONVEX', rotate_method='CARDINAL')
    finally:
        bpy.ops.object.mode_set(mode='OBJECT')
    for uv in mesh.uv_layers.active.uv:
        uv.vector *= surface_fraction


def texel_size(obj, resolution):
    """World length of one texel, averaged over the whole UV layout."""
    mesh = obj.data
    layer = mesh.uv_layers.active.uv
    world = uv_area = 0.0
    for polygon in mesh.polygons:
        world += polygon.area
        points = [layer[i].vector for i in polygon.loop_indices]
        uv_area += abs(sum(a.x * b.y - b.x * a.y
                           for a, b in zip(points, points[1:] + points[:1]))) / 2
    if uv_area <= 0:
        return 0.0
    return math.sqrt(world / uv_area) / resolution


def validate(obj, require_uv, repeating=()):
    """repeating: material indices whose textures tile, so their UVs may exceed 0..1."""
    mesh = obj.data
    if not mesh.polygons or mesh.validate(verbose=False, clean_customdata=False):
        raise ValueError("Mesh validation failed.")
    if require_uv:
        if len(mesh.uv_layers) != 1:
            raise ValueError("Exactly one game UV map is required.")
        layer = mesh.uv_layers.active.uv
        for polygon in mesh.polygons:
            tiled = polygon.material_index in repeating
            for index in polygon.loop_indices:
                for v in layer[index].vector:
                    if not math.isfinite(v) or (not tiled and not -.00001 <= v <= 1.00001):
                        raise ValueError("UV coordinates are outside the atlas.")

