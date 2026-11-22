"""Compare projected silhouettes from several orthographic viewpoints."""
from mathutils import Vector
from mathutils.bvhtree import BVHTree
import numpy as np


VIEWS = ((1, 0, 0), (0, 1, 0), (0, 0, 1),
         (1, 1, 1), (1, -1, 1), (-1, 1, 1))


def surface(sources):
    vertices, faces = [], []
    for source in sources:
        offset = len(vertices)
        vertices.extend(v.co for v in source.data.vertices)
        faces.extend([offset + index for index in face.vertices]
                     for face in source.data.polygons)
    return BVHTree.FromPolygons(vertices, faces)


def compare(source, candidate, points, transform=None, resolution=40):
    bounds = np.asarray(points)
    center = bounds.mean(axis=0)
    reach = np.linalg.norm(bounds-center, axis=1).max() * 2
    scores = []
    for direction in VIEWS:
        view = Vector(direction).normalized()
        up = Vector((0, 0, 1)) if abs(view.z) < .95 else Vector((0, 1, 0))
        right = up.cross(view).normalized()
        up = view.cross(right)
        horizontal, vertical = bounds @ np.asarray(right), bounds @ np.asarray(up)
        step = max(np.ptp(horizontal), np.ptp(vertical)) / resolution
        depth = center @ np.asarray(view) + reach
        base = depth * view
        ray = -view
        overlap = union = 0
        for x in np.arange(horizontal.min() + step/2, horizontal.max(), step):
            for y in np.arange(vertical.min() + step/2, vertical.max(), step):
                origin = base + x*right + y*up
                a = source.ray_cast(origin, ray, reach*2)[0] is not None
                start, heading, distance = (origin, ray, reach*2)
                if transform:
                    start, heading, distance = transform(start, heading, distance)
                b = candidate.ray_cast(start, heading, distance)[0] is not None
                overlap += a and b
                union += a or b
        if union:
            scores.append(overlap / union)
    return scores

