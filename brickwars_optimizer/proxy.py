"""Rebuild an elongated source as a closed cylinder with a three-island UV map."""
import math

import bpy
import numpy as np
from mathutils.bvhtree import BVHTree

from . import silhouette


def radial_profile(tree, basis, center, low, high, radius):
    """Measure the outer surface along the axis, then simplify its profile."""
    angles = np.arange(24) * (2 * math.pi / 24)
    directions = np.column_stack((np.cos(angles), np.sin(angles), np.zeros(24))) @ basis.T
    length = high - low
    stations = np.linspace(low, high, 97)
    probe = np.clip(stations, low + length * .0002, high - length * .0002)
    origin_radius = radius * 1.05
    radii = []
    for z in probe:
        axis_point = np.array((*center, z)) @ basis.T
        hits = []
        for direction in directions:
            point, _, _, distance = tree.ray_cast(
                axis_point + direction * origin_radius, -direction, origin_radius)
            if point is not None:
                hits.append(origin_radius - distance)
        radii.append(max(hits) if len(hits) >= 8 else float('nan'))
    radii = np.asarray(radii)
    valid = np.isfinite(radii)
    if valid.sum() < len(stations) * .8:
        return [(low, radius), (high, radius)]
    radii = np.interp(stations, stations[valid], radii[valid])
    padded = np.pad(radii, 1, mode='edge')
    radii = np.median(np.lib.stride_tricks.sliding_window_view(padded, 3), axis=1)
    if np.ptp(radii) <= radius * .08:
        return [(low, radius), (high, radius)]

    tolerance = radius * .07
    while True:
        indices = {0, len(stations)-1}
        intervals = [(0, len(stations)-1)]
        while intervals:
            first, last = intervals.pop()
            if last - first <= 1:
                continue
            linear = np.linspace(radii[first], radii[last], last-first+1)
            errors = np.abs(radii[first:last+1] - linear)
            inner = int(np.argmax(errors))
            if errors[inner] > tolerance:
                middle = first + inner
                indices.add(middle)
                intervals.extend(((first, middle), (middle, last)))
        if len(indices) <= 12:
            break
        tolerance *= 1.25

    profile = [(stations[i], max(radii[i], radius * .025)) for i in sorted(indices)]
    for i in range(1, len(profile)-2):
        z0, r0 = profile[i]
        z1, r1 = profile[i+1]
        if 0 < z1-z0 <= length / 80 and abs(r0-r1) > radius * .16:
            shoulder = (z0+z1) / 2
            profile[i:i+2] = [(shoulder, r0), (shoulder, r1)]
    return profile


def cylinder(sources, sides, generate_uv, margin):
    # Unique positions keep duplicated import seams from biasing the fit.
    points = np.unique(np.concatenate([
        np.array([tuple(v.co) for v in source.data.vertices]) for source in sources
        if source.data.vertices
    ]), axis=0)
    if len(points) < 4:
        raise ValueError("A cylinder needs a three-dimensional source mesh.")
    centered = points - points.mean(axis=0)
    axis = np.linalg.eigh(centered.T @ centered)[1][:, -1]
    dominant = np.argmax(np.abs(axis))
    if axis[dominant] < 0:
        axis = -axis
    reference = np.eye(3)[np.argmin(np.abs(axis))]
    tangent = np.cross(reference, axis)
    tangent /= np.linalg.norm(tangent)
    bitangent = np.cross(axis, tangent)
    basis = np.column_stack((tangent, bitangent, axis))
    local = points @ basis
    low, high = local.min(axis=0), local.max(axis=0)
    center = (low[:2] + high[:2]) / 2
    radius = np.linalg.norm(local[:, :2] - center, axis=1).max()
    spans = high - low
    if spans.max() <= 0 or spans.min() < spans.max() * 1e-7:
        raise ValueError("Cannot fit a cylinder to a flat or degenerate source.")
    tree = silhouette.surface(sources)
    profile = radial_profile(tree, basis, center, low[2], high[2], radius)
    factor = 1 / math.cos(math.pi / sides)
    profile = [(z, r*factor) for z, r in profile]
    outer = max(r for _, r in profile)
    angles = np.arange(sides) * (2 * math.pi / sides)
    ring = np.column_stack((np.cos(angles), np.sin(angles)))
    vertices = [(x, y, z) for z, r in profile for x, y in center + r*ring]
    faces, coordinates = [], []
    distances = np.r_[0, np.cumsum(np.linalg.norm(np.diff(profile, axis=0), axis=1))]
    circumference = sides * 2 * outer * math.sin(math.pi / sides)
    scale = min((1 - 2 * margin) / circumference,
                (1 - 3 * margin) / (distances[-1] + 2 * outer))
    u0, v0 = margin, 2 * margin + 2 * outer * scale
    for band in range(len(profile)-1):
        offset = band * sides
        bottom = v0 + distances[band] * scale
        top = v0 + distances[band+1] * scale
        for i in range(sides):
            j = (i + 1) % sides
            left = u0 + i * circumference * scale / sides
            right = u0 + (i + 1) * circumference * scale / sides
            faces.extend([(offset+i, offset+j, offset+j+sides),
                          (offset+i, offset+j+sides, offset+i+sides)])
            coordinates.extend([[(left, bottom), (right, bottom), (right, top)],
                                [(left, bottom), (right, top), (left, top)]])
    wall_faces = len(faces)
    for cap, index in enumerate((0, len(profile)-1)):
        offset = index * sides
        cap_uv = ring * (profile[index][1] * scale) + (
            (2 * cap + 1) * outer * scale + (cap + 1) * margin,
            outer * scale + margin,
        )
        for i in range(1, sides - 1):
            order = (0, i + 1, i) if cap == 0 else (0, i, i + 1)
            faces.append(tuple(offset + j for j in order))
            coordinates.append([cap_uv[j] for j in order])
    world_vertices = np.asarray(vertices) @ basis.T
    candidate = BVHTree.FromPolygons(world_vertices, faces, all_triangles=True)
    bounds = np.vstack((points.min(axis=0), points.max(axis=0),
                        world_vertices.min(axis=0), world_vertices.max(axis=0)))
    scores = silhouette.compare(tree, candidate, bounds)
    average = sum(scores) / len(scores) if scores else 0
    if average < .62 or min(scores) < .40:
        raise ValueError(f"Cylinder rejected: silhouette overlap is {average:.0%}. Use a method that preserves the separate shapes.")
    result = bpy.data.meshes.new("BrickWars Profile Cylinder")
    result.from_pydata(world_vertices, [], faces)
    result.update()
    for polygon in result.polygons[:wall_faces]:
        band = polygon.index // (2*sides)
        polygon.use_smooth = bool(profile[band][0] != profile[band+1][0])
    if generate_uv:
        layer = result.uv_layers.new(name="BrickWars UV")
        layer.uv.foreach_set("vector", np.asarray(coordinates, dtype=np.float32).ravel())
    return result, outer * 1.15


