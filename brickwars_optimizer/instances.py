"""Find identical LEGO parts so they can share one region of the texture atlas.

Two parts match when they have the same material colors and printed images and
one maps onto the other by a rigid rotation and translation. Candidate rotations
come from each part's principal axes and dominant face normals combined with the
24 axis permutations; the winner is verified point by point in both directions.
Mirrored parts (left and right wings) never match because only proper rotations
are tried, so normal maps are never flipped.
"""
import itertools
import math
from collections import defaultdict

import numpy as np
from mathutils import Vector
from mathutils.bvhtree import BVHTree

from .rebuild import material_color

UP_COS = math.cos(math.radians(5))
SAMPLE_POINTS = 96
TOLERANCE = 2e-3


def _rotations():
    result = []
    for order in itertools.permutations(range(3)):
        for signs in itertools.product((1, -1), repeat=3):
            matrix = np.zeros((3, 3))
            for row, (column, sign) in enumerate(zip(order, signs)):
                matrix[row, column] = sign
            if np.linalg.det(matrix) > 0:
                result.append(matrix)
    return result


ROTATIONS = _rotations()
IDENTITY = (np.eye(3), np.zeros(3))


def material_key(data):
    """Materials that look the same give the same key, whatever their names."""
    key = []
    for material in data.materials:
        if material is None:
            key.append(None)
            continue
        images, alpha = (), 1.0
        tree = material.node_tree
        if tree:
            images = tuple(sorted(n.image.name for n in tree.nodes
                                  if n.type == 'TEX_IMAGE' and n.image))
            shader = next((n for n in tree.nodes if n.type == 'BSDF_PRINCIPLED'), None)
            if shader:
                alpha = shader.inputs['Alpha'].default_value
        color = tuple(round(c, 3) for c in material_color(material))
        key.append((color, round(alpha, 3), images))
    return tuple(key)


class Shape:
    __slots__ = ("key", "co", "center", "spread", "area", "size", "frames", "sample", "_tree",
                 "tris")

    def tree(self):
        if self._tree is None:
            self._tree = BVHTree.FromPolygons([tuple(p) for p in self.co],
                                              self.tris.tolist(), all_triangles=True)
        return self._tree


def describe(source):
    from .parts import normal_frame, triangles_of
    data = source.data
    co, normals, areas, _ = triangles_of(data)
    if not len(areas):
        return None
    shape = Shape()
    tris = np.empty(len(data.loop_triangles) * 3, dtype=np.int64)
    data.loop_triangles.foreach_get("vertices", tris)
    shape.tris = tris.reshape(-1, 3)
    shape.key = (len(data.vertices), len(data.loop_triangles), material_key(data))
    shape.co = co
    shape.center = co.mean(axis=0)
    centered = co - shape.center
    values, vectors = np.linalg.eigh(centered.T @ centered / len(co))
    principal = vectors.T[::-1].copy()
    if np.linalg.det(principal) < 0:
        principal[2] *= -1
    frame = normal_frame(co, normals, areas)
    if np.linalg.det(frame) < 0:
        frame = frame * np.array([[1], [1], [-1]])
    shape.frames = (principal, frame)
    shape.spread = values[::-1]
    shape.area = float(areas.sum())
    shape.size = float(np.ptp(co, axis=0).max())
    step = max(1, len(co) // SAMPLE_POINTS)
    shape.sample = co[::step]
    shape._tree = None
    return shape


def _fits(points, target, limit):
    tree = target.tree()
    for p in points:
        hit = tree.find_nearest(Vector(p), limit)
        if hit[0] is None:
            return False
    return True


def match(shape, reference, upright):
    """Rigid (R, t) with R @ x + t mapping shape onto reference, or None."""
    if abs(shape.area - reference.area) > 1e-3 * reference.area:
        return None
    if np.any(np.abs(shape.spread - reference.spread) > 1e-3 * reference.spread[0] + 1e-12):
        return None
    limit = TOLERANCE * reference.size
    tried = []
    for own, other in zip(shape.frames, reference.frames):
        for flip in ROTATIONS:
            rotation = other.T @ flip @ own
            if upright and rotation[2, 2] < UP_COS:
                continue
            if any(np.abs(rotation - r).max() < 1e-6 for r in tried):
                continue
            tried.append(rotation)
            offset = reference.center - rotation @ shape.center
            if not _fits(shape.sample @ rotation.T + offset, reference, limit):
                continue
            back = (reference.sample - offset) @ rotation
            if _fits(back, shape, limit):
                return rotation, offset
    return None


def classify(sources, share, upright):
    """Generator yielding progress; returns (leader index, transform) per source.

    The transform maps the part onto its leader's position: leader = R @ part + t.
    """
    leader = list(range(len(sources)))
    transform = [IDENTITY] * len(sources)
    if not share:
        return leader, transform
    shapes = []
    buckets = defaultdict(list)
    for number, source in enumerate(sources):
        shape = describe(source)
        shapes.append(shape)
        if shape is not None:
            for candidate in buckets[shape.key]:
                found = match(shape, shapes[candidate], upright)
                if found:
                    leader[number], transform[number] = candidate, found
                    break
            else:
                buckets[shape.key].append(number)
        if number % 16 == 0:
            yield number / len(sources)
    return leader, transform
