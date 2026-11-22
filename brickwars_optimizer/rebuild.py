"""Rebuild a dense brick model from a few convex blocks, like a manual low-poly pass.

The model is voxelized once. The rebuild starts with one box around everything and
greedily applies whichever operation removes the most empty volume per added
triangle: splitting a block at its best plane, or cutting a 45 degree chamfer.
Every block encloses its share of the surface, so baking always projects inward.
"""
import heapq
import itertools
import math

import bmesh
import bpy
import numpy as np
from mathutils.bvhtree import BVHTree

IDENTITY = np.eye(3)
FACE_DIRS = np.array([s * IDENTITY[a] for a in range(3) for s in (1, -1)])
CHAMFER_DIRS = np.array(
    [v for v in itertools.product((-1, 0, 1), repeat=3) if sum(map(abs, v)) >= 2],
    dtype=float)
CHAMFER_DIRS /= np.linalg.norm(CHAMFER_DIRS, axis=1)[:, None]
_golden = np.pi * (3 - math.sqrt(5))
_z = np.linspace(1 - 1 / 230, 1 / 230 - 1, 230)
_r = np.sqrt(1 - _z ** 2)
SUPPORT_DIRS = np.vstack((
    FACE_DIRS, CHAMFER_DIRS,
    np.stack((_r * np.cos(_golden * np.arange(230)), _r * np.sin(_golden * np.arange(230)), _z),
             axis=1)))
SPLIT_DIRS = np.vstack((IDENTITY, np.array(
    [(1, 1, 0), (1, -1, 0), (1, 0, 1), (1, 0, -1), (0, 1, 1), (0, 1, -1)]) / math.sqrt(2)))
COMBOS = {}
AREA_WEIGHT = 4


# --------------------------------------------------------------------------- input

def gather(sources):
    """Return all source vertices and triangles in the shared snapshot space."""
    verts, tris, offset = [], [], 0
    for obj in sources:
        data = obj.data
        data.calc_loop_triangles()
        co = np.empty(len(data.vertices) * 3, dtype=np.float64)
        data.vertices.foreach_get("co", co)
        tri = np.empty(len(data.loop_triangles) * 3, dtype=np.int64)
        data.loop_triangles.foreach_get("vertices", tri)
        verts.append(co.reshape(-1, 3))
        tris.append(tri.reshape(-1, 3) + offset)
        offset += len(data.vertices)
    return np.concatenate(verts), np.concatenate(tris)


def sample_surface(verts, tris, spacing):
    """Points on every triangle no further apart than spacing, deduplicated."""
    a, b, c = (verts[tris[:, i]] for i in range(3))
    longest = np.maximum.reduce([np.linalg.norm(b - a, axis=1),
                                 np.linalg.norm(c - b, axis=1),
                                 np.linalg.norm(a - c, axis=1)])
    steps = np.clip(np.ceil(longest / spacing), 1, 96).astype(int)
    points = [verts[np.unique(tris)]]
    for k in np.unique(steps):
        if k == 1:
            continue
        sel = steps == k
        grid = np.array([(i, j) for i in range(k + 1) for j in range(k + 1 - i)], float) / k
        u, v = grid[:, 0], grid[:, 1]
        w = 1 - u - v
        pts = (a[sel, None] * w[:, None] + b[sel, None] * u[:, None] + c[sel, None] * v[:, None])
        points.append(pts.reshape(-1, 3))
    points = np.concatenate(points)
    keys = np.floor(points / (spacing * .5)).astype(np.int64)
    keys -= keys.min(axis=0)
    span = keys.max(axis=0) + 1
    packed = (keys[:, 0] * span[1] + keys[:, 1]) * span[2] + keys[:, 2]
    _, first = np.unique(packed, return_index=True)
    return points[first]


# --------------------------------------------------------------------------- grid

def _dilate(mask):
    out = mask.copy()
    out[1:] |= mask[:-1]
    out[:-1] |= mask[1:]
    out[:, 1:] |= mask[:, :-1]
    out[:, :-1] |= mask[:, 1:]
    out[:, :, 1:] |= mask[:, :, :-1]
    out[:, :, :-1] |= mask[:, :, 1:]
    return out


def _exterior(blocked):
    reach = np.zeros_like(blocked)
    reach[0], reach[-1] = True, True
    reach[:, 0], reach[:, -1] = True, True
    reach[:, :, 0], reach[:, :, -1] = True, True
    reach &= ~blocked
    while True:
        before = int(reach.sum())
        for _ in range(8):
            reach = _dilate(reach) & ~blocked
        if int(reach.sum()) == before:
            return reach


class Grid:
    """Solid occupancy with gaps between bricks closed and cavities filled."""

    def __init__(self, points, resolution):
        low, high = points.min(axis=0), points.max(axis=0)
        self.size = float(max(high - low))
        self.h = self.size / resolution
        pad = 3
        self.origin = low - pad * self.h
        self.shape = tuple(np.ceil((high - low) / self.h).astype(int) + 2 * pad + 1)
        surface = np.zeros(self.shape, dtype=bool)
        index = np.floor((points - self.origin) / self.h).astype(int)
        surface[index[:, 0], index[:, 1], index[:, 2]] = True
        thick = _dilate(surface)
        filled = ~_exterior(thick)
        eroded = ~_dilate(~filled)
        self.solid = eroded | surface
        self.table = self._summed((~self.solid).astype(np.int32))
        # Voxels more than one cell away from the model: a block face over these
        # is visibly wrong no matter how thin the block is.
        self.far = ~_dilate(self.solid)
        self.far_table = self._summed(self.far.astype(np.int32))
        self.solid_volume = int(self.solid.sum())

    def _summed(self, values):
        table = np.zeros(tuple(s + 1 for s in self.shape), dtype=np.int32)
        table[1:, 1:, 1:] = values.cumsum(0).cumsum(1).cumsum(2)
        return table

    @staticmethod
    def _count(t, s, e):
        x0, y0, z0 = s[..., 0], s[..., 1], s[..., 2]
        x1, y1, z1 = e[..., 0], e[..., 1], e[..., 2]
        return (t[x1, y1, z1] - t[x0, y1, z1] - t[x1, y0, z1] - t[x1, y1, z0]
                + t[x0, y0, z1] + t[x0, y1, z0] + t[x1, y0, z0] - t[x0, y0, z0])

    def cover_range(self, low, high):
        """Voxel index ranges [start, stop) touched by [low, high]."""
        limit = np.array(self.shape)
        start = np.floor((low - self.origin) / self.h).astype(int)
        stop = np.floor((high - self.origin) / self.h).astype(int) + 1
        return np.clip(start, 0, limit - 1), np.clip(stop, 1, limit)

    def far_on_box_faces(self, low, high):
        """Far voxels touched by the six faces of many axis-aligned boxes."""
        s, e = self.cover_range(low, high)
        total = 0
        for axis in range(3):
            for layer in (s[..., axis], e[..., axis] - 1):
                a, b = s.copy(), e.copy()
                a[..., axis] = layer
                b[..., axis] = layer + 1
                total = total + self._count(self.far_table, a, b)
        return total

    def cost_boxes(self, low, high):
        """Empty volume plus weighted far face area for axis-aligned boxes."""
        return self.empty_in_boxes(low, high) + AREA_WEIGHT * self.far_on_box_faces(low, high)

    def cost_polytope(self, normals, offsets, corners):
        if len(corners) < 4:
            return 0
        cost = self.empty_in_polytope(normals, offsets, corners)
        s, e = self.cover_range(corners.min(axis=0), corners.max(axis=0))
        cells = np.argwhere(self.far[s[0]:e[0], s[1]:e[1], s[2]:e[2]])
        if len(cells):
            centers = self.origin + (cells + s + .5) * self.h
            distance = centers @ normals.T - offsets
            shell = np.all(distance <= .5 * self.h, axis=1) & (distance.max(axis=1) >= -self.h)
            cost += AREA_WEIGHT * int(np.count_nonzero(shell))
        return cost

    def index_range(self, low, high):
        """Voxel index ranges [start, stop) whose centers lie inside [low, high]."""
        start = np.ceil((low - self.origin) / self.h - .5).astype(int)
        stop = np.floor((high - self.origin) / self.h - .5).astype(int) + 1
        limit = np.array(self.shape)
        return np.clip(start, 0, limit), np.clip(stop, 0, limit)

    def empty_in_boxes(self, low, high):
        """Count empty voxels inside many axis-aligned boxes at once."""
        s, e = self.index_range(low, high)
        e = np.maximum(e, s)
        t = self.table
        x0, y0, z0 = s[..., 0], s[..., 1], s[..., 2]
        x1, y1, z1 = e[..., 0], e[..., 1], e[..., 2]
        return (t[x1, y1, z1] - t[x0, y1, z1] - t[x1, y0, z1] - t[x1, y1, z0]
                + t[x0, y0, z1] + t[x0, y1, z0] + t[x1, y0, z0] - t[x0, y0, z0])

    def empty_in_polytope(self, normals, offsets, corners):
        """Empty volume in voxels: analytic volume minus solid voxel centers inside."""
        if len(corners) < 4:
            return 0
        volume = polytope_volume(normals, offsets, corners, self.size * 1e-6) / self.h ** 3
        s, e = self.index_range(corners.min(axis=0), corners.max(axis=0))
        if np.any(e <= s):
            return volume
        cells = np.argwhere(self.solid[s[0]:e[0], s[1]:e[1], s[2]:e[2]])
        if not len(cells):
            return volume
        centers = self.origin + (cells + s + .5) * self.h
        inside = np.all(centers @ normals.T <= offsets + 1e-9 * self.size, axis=1)
        return max(0.0, volume - int(np.count_nonzero(inside)))


# --------------------------------------------------------------------------- polytopes

def polytope_vertices(normals, offsets, tolerance):
    k = len(normals)
    if k not in COMBOS:
        COMBOS[k] = np.array(list(itertools.combinations(range(k), 3)))
    combos = COMBOS[k]
    a = normals[combos]
    det = np.linalg.det(a)
    ok = np.abs(det) > 1e-9
    points = np.linalg.solve(a[ok], offsets[combos[ok]][..., None])[..., 0]
    feasible = np.all(points @ normals.T <= offsets + tolerance, axis=1)
    points = points[feasible]
    if not len(points):
        return points
    keys = np.round(points / tolerance).astype(np.int64)
    _, first = np.unique(keys, axis=0, return_index=True)
    return points[np.sort(first)]


BOX_SIGNS = np.array(list(itertools.product((0, 1), repeat=3)))


def box_corners(frame, offsets):
    """Eight corners of a box given in FACE_DIRS order within frame."""
    high, low = offsets[0::2], -offsets[1::2]
    return np.where(BOX_SIGNS, high, low) @ frame


def polytope_volume(normals, offsets, corners, tolerance):
    """Sum of face pyramids: V = 1/3 * sum(area_f * offset_f)."""
    if len(normals) == 6 and np.allclose(normals[0::2], -normals[1::2]):
        return float(np.prod(offsets[0::2] + offsets[1::2]))
    volume = 0.0
    distance = corners @ normals.T - offsets
    for f, normal in enumerate(normals):
        face = corners[np.abs(distance[:, f]) < 10 * tolerance]
        if len(face) < 3:
            continue
        center = face.mean(axis=0)
        u = face[0] - center
        u /= np.linalg.norm(u) or 1
        v = np.cross(normal, u)
        rel = face - center
        x, y = rel @ u, rel @ v
        order = np.argsort(np.arctan2(y, x))
        x, y = x[order], y[order]
        area = .5 * abs(np.dot(x, np.roll(y, -1)) - np.dot(y, np.roll(x, -1)))
        volume += area * offsets[f] / 3
    return volume


def triangle_count(vertex_count):
    return max(0, 2 * vertex_count - 4)


class Block:
    __slots__ = ("id", "index", "frame", "dirs", "overrides", "normals", "offsets",
                 "corners", "low", "high", "error", "tris", "alive", "support", "moments")


class Rebuilder:
    """Split finely, merge neighbours into oriented boxes, then spend the rest."""

    def __init__(self, points, grid, budget, chamfers):
        self.points = points
        self.grid = grid
        self.budget = budget
        self.chamfers = chamfers
        self.tolerance = grid.size * 1e-6
        self.min_gain = 3
        self.blocks = {}
        self.ids = itertools.count()
        self.counter = itertools.count()
        self.total = 0

    # -- evaluation ---------------------------------------------------------
    def planes(self, index, frame, dirs, overrides):
        normals = dirs @ frame
        offsets = (self.points[index] @ normals.T).max(axis=0)
        for j, value in overrides.items():
            offsets[j] = max(offsets[j], value)
        return normals, offsets

    def create(self, index, frame, dirs=FACE_DIRS, overrides=None):
        block = Block()
        block.id = next(self.ids)
        block.index, block.frame, block.dirs = index, frame, dirs
        block.overrides = overrides or {}
        block.normals, block.offsets = self.planes(index, frame, dirs, block.overrides)
        if len(dirs) == 6:
            # Flat decals and single faces still need a closed, non-degenerate box.
            deficit = np.maximum(0, .25 * self.grid.h - block.offsets[0::2] - block.offsets[1::2])
            block.offsets[0::2] += deficit / 2
            block.offsets[1::2] += deficit / 2
            block.corners = box_corners(frame, block.offsets)
        else:
            block.corners = polytope_vertices(block.normals, block.offsets, self.tolerance)
        block.low, block.high = block.corners.min(axis=0), block.corners.max(axis=0)
        block.tris = triangle_count(len(block.corners))
        if frame is IDENTITY and len(dirs) == 6:
            block.error = int(self.grid.cost_boxes(-block.offsets[1::2], block.offsets[0::2]))
        else:
            block.error = self.grid.cost_polytope(block.normals, block.offsets, block.corners)
        block.support = block.moments = None
        block.alive = True
        self.blocks[block.id] = block
        self.total += block.tris
        return block

    def remove(self, block):
        block.alive = False
        del self.blocks[block.id]
        self.total -= block.tris

    def box_error(self, points, frame):
        """Empty volume of the tight box around points in the given frame."""
        if frame is IDENTITY:
            return int(self.grid.cost_boxes(points.min(axis=0), points.max(axis=0)))
        normals = FACE_DIRS @ frame
        offsets = (points @ normals.T).max(axis=0)
        return self.grid.cost_polytope(normals, offsets, box_corners(frame, offsets))

    def oriented_box_error(self, frame, low, high):
        normals = FACE_DIRS @ frame
        offsets = np.array([high[0], -low[0], high[1], -low[1], high[2], -low[2]])
        return self.grid.cost_polytope(normals, offsets, box_corners(frame, offsets))

    @staticmethod
    def principal_frame(moments):
        count, total, outer = moments
        mean = total / count
        _, vectors = np.linalg.eigh(outer / count - np.outer(mean, mean))
        frame = vectors.T[::-1].copy()
        if np.linalg.det(frame) < 0:
            frame[2] *= -1
        return frame

    def best_frame(self, points, frames, error=None):
        """Pick the frame whose tight box leaves the least empty volume."""
        best_error = self.box_error(points, IDENTITY) if error is None else error
        best = IDENTITY
        seen = [IDENTITY]
        for frame in frames:
            if any(np.abs(frame @ other.T).max(axis=1).min() > .9998 for other in seen):
                continue
            seen.append(frame)
            error = self.box_error(points, frame)
            if error < .8 * best_error:
                best, best_error = frame, error
        return best, best_error

    # -- split --------------------------------------------------------------
    def best_split(self, block, force=False):
        """Best plane split; with force, fall back to halving the longest side."""
        index = block.index
        if len(index) < 8 or block.error < self.min_gain:
            return None
        frame = block.frame
        q_all = self.points[index] @ frame.T
        # A regular subsample is enough to choose the plane; all points are split.
        q = q_all
        if len(index) > 15000:
            q = q_all[np.linspace(0, len(index) - 1, 15000).astype(int)]
        h = self.grid.h
        directions = SPLIT_DIRS if frame is IDENTITY else IDENTITY
        limit = 384 if frame is IDENTITY else 12
        best = None
        for axis, direction in enumerate(directions):
            projection = q @ direction
            order = np.argsort(projection, kind='stable')
            s = q[order]
            p = projection[order]
            gaps = np.diff(p)
            cuts = np.nonzero(gaps > 1e-12)[0] + 1
            if not len(cuts):
                continue
            if len(cuts) > limit:
                even = cuts[np.linspace(0, len(cuts) - 1, limit).astype(int)]
                wide = cuts[np.argsort(gaps[cuts - 1])[-max(4, limit // 4):]]
                cuts = np.unique(np.concatenate((even, wide)))
            lo_pre = np.minimum.accumulate(s, axis=0)
            hi_pre = np.maximum.accumulate(s, axis=0)
            lo_suf = np.minimum.accumulate(s[::-1], axis=0)[::-1]
            hi_suf = np.maximum.accumulate(s[::-1], axis=0)[::-1]
            left_lo, left_hi = lo_pre[cuts - 1].copy(), hi_pre[cuts - 1].copy()
            right_lo, right_hi = lo_suf[cuts].copy(), hi_suf[cuts].copy()
            middle = (p[cuts] + p[cuts - 1]) / 2
            joined = np.zeros(len(cuts), dtype=bool)
            if axis < 3:
                joined = (p[cuts] - p[cuts - 1]) < 1.5 * h
                left_hi[joined, axis] = middle[joined]
                right_lo[joined, axis] = middle[joined]
            if frame is IDENTITY:
                errors = (self.grid.cost_boxes(left_lo, left_hi)
                          + self.grid.cost_boxes(right_lo, right_hi))
            else:
                errors = np.array([
                    self.oriented_box_error(frame, a, b) + self.oriented_box_error(frame, c, d)
                    for a, b, c, d in zip(left_lo, left_hi, right_lo, right_hi)])
            pick = int(np.argmin(errors))
            gain = block.error - int(errors[pick])
            if best is None or gain > best[0]:
                best = (gain, axis, direction, float(middle[pick]), bool(joined[pick]))
        if best is None or best[0] < self.min_gain:
            if not force:
                return None
            extent = q_all.max(axis=0) - q_all.min(axis=0)
            axis = int(np.argmax(extent))
            best = (0, axis, IDENTITY[axis], float(np.median(q_all[:, axis])), True)
        gain, axis, direction, middle, joined = best
        left = q_all @ direction <= middle
        if left.all() or not left.any():
            return None
        return gain, 24 - block.tris, ('split', axis, index[left], index[~left], joined, middle)

    def split(self, block, op, reframe):
        _, axis, left, right, joined, middle = op
        self.remove(block)
        frame = block.frame
        left_over, right_over = dict(block.overrides), dict(block.overrides)
        if joined:
            left_over[2 * axis] = middle
            right_over[2 * axis + 1] = -middle
        children = []
        for index, overrides in ((left, left_over), (right, right_over)):
            overrides = self.clip(index, frame, overrides)
            child_frame = frame
            if reframe and len(index) >= 24:
                samples = self.points[index]
                error = self.box_error(samples, frame)
                if error >= 64:
                    moments = (len(samples), samples.sum(axis=0), samples.T @ samples)
                    child_frame, _ = self.best_frame(
                        samples, [frame, self.principal_frame(moments)],
                        error if frame is IDENTITY else None)
            if child_frame is not frame:
                overrides = {}
            children.append(self.create(index, child_frame, overrides=overrides))
        return children

    def clip(self, index, frame, overrides):
        """Drop inherited joins that no longer touch this block's own samples."""
        if not overrides:
            return overrides
        q = self.points[index] @ frame.T
        kept = {}
        for j, value in overrides.items():
            sign = 1 if j % 2 == 0 else -1
            extreme = (sign * q[:, j // 2]).max()
            if 0 <= value - extreme < 1.5 * self.grid.h:
                kept[j] = value
        return kept

    # -- chamfer ------------------------------------------------------------
    def best_chamfer(self, block):
        if not self.chamfers or block.error < self.min_gain or len(block.dirs) >= 26:
            return None
        best = None
        present = {tuple(np.round(d, 6)) for d in block.dirs}
        samples = self.points[block.index]
        for direction in CHAMFER_DIRS:
            if tuple(np.round(direction, 6)) in present:
                continue
            normal = direction @ block.frame
            offset = float((samples @ normal).max())
            normals = np.vstack((block.normals, normal))
            offsets = np.append(block.offsets, offset)
            corners = polytope_vertices(normals, offsets, self.tolerance)
            added = triangle_count(len(corners)) - block.tris
            if added <= 0:
                continue
            gain = block.error - self.grid.cost_polytope(normals, offsets, corners)
            if gain < self.min_gain:
                continue
            if best is None or gain / added > best[0] / best[1]:
                best = (gain, added, ('chamfer', direction))
        return best

    # -- merge --------------------------------------------------------------
    def touching(self, a, b):
        margin = 2 * self.grid.h
        return bool(np.all(a.low <= b.high + margin) and np.all(b.low <= a.high + margin))

    @staticmethod
    def summarize(block, samples):
        """Extreme points in many directions and second moments of the samples."""
        block.support = samples[np.unique((samples @ SUPPORT_DIRS.T).argmax(axis=0))]
        block.moments = (len(samples), samples.sum(axis=0), samples.T @ samples)

    def merge_option(self, a, b):
        # Support points and moments stand in for all samples of the union.
        points = np.vstack((a.support, b.support))
        identity = self.box_error(points, IDENTITY)
        frames = [a.frame, b.frame]
        if identity > a.error + b.error + 16:
            moments = tuple(x + y for x, y in zip(a.moments, b.moments))
            frames.append(self.principal_frame(moments))
        frame, error = self.best_frame(points, frames, identity)
        saved = a.tris + b.tris - 12
        increase = error - a.error - b.error
        return increase, saved, frame

    def merge(self, stop):
        blocks = list(self.blocks.values())
        for block in blocks:
            self.summarize(block, self.points[block.index])
        lows = np.array([b.low for b in blocks])
        highs = np.array([b.high for b in blocks])
        margin = 2 * self.grid.h
        neighbours = {b.id: set() for b in blocks}
        for i, block in enumerate(blocks):
            hits = np.nonzero(np.all(lows[i + 1:] <= block.high + margin, axis=1)
                              & np.all(highs[i + 1:] >= block.low - margin, axis=1))[0]
            for j in hits + i + 1:
                neighbours[block.id].add(blocks[j].id)
                neighbours[blocks[j].id].add(block.id)
        heap = []

        def offer(a, b):
            increase, saved, frame = self.merge_option(a, b)
            if saved > 0:
                heapq.heappush(heap, (increase / saved, next(self.counter), a, b, frame, increase))

        for block in blocks:
            for other in neighbours[block.id]:
                if other > block.id:
                    offer(block, self.blocks[other])
        steps = 0
        while heap:
            _, _, a, b, frame, increase = heapq.heappop(heap)
            if not (a.alive and b.alive):
                continue
            if self.total <= stop and increase > self.min_gain:
                break
            self.remove(a)
            self.remove(b)
            merged = self.create(np.concatenate((a.index, b.index)), frame)
            self.summarize(merged, np.vstack((a.support, b.support)))
            merged.moments = tuple(x + y for x, y in zip(a.moments, b.moments))
            near = (neighbours.pop(a.id) | neighbours.pop(b.id)) - {a.id, b.id}
            neighbours[merged.id] = set()
            for other_id in near:
                neighbours[other_id] -= {a.id, b.id}
                other = self.blocks[other_id]
                if self.touching(merged, other):
                    neighbours[merged.id].add(other_id)
                    neighbours[other_id].add(merged.id)
                    offer(merged, other)
            steps += 1
            if steps % 16 == 0:
                yield steps

    # -- driver -------------------------------------------------------------
    def refine(self, limit, reframe, splits_only=False):
        heap = []

        def offer(block):
            # Over-segmentation splits the worst block even without immediate gain,
            # because X- or L-shaped clusters need several cuts before boxes shrink.
            options = [self.best_split(block, force=splits_only)]
            if not splits_only:
                options.append(self.best_chamfer(block))
            for option in options:
                if option:
                    gain, added, op = option
                    key = -block.error if splits_only else -gain / max(added, 1)
                    heapq.heappush(heap, (key, next(self.counter), block, gain, added, op))

        for block in list(self.blocks.values()):
            offer(block)
        steps = 0
        while heap:
            _, _, block, gain, added, op = heapq.heappop(heap)
            if not block.alive:
                continue
            if self.total + added > limit:
                if splits_only:
                    break
                continue
            if op[0] == 'split':
                for child in self.split(block, op, reframe):
                    offer(child)
            else:
                self.remove(block)
                offer(self.create(block.index, block.frame,
                                  np.vstack((block.dirs, op[1])), block.overrides))
            steps += 1
            if steps % 16 == 0:
                yield steps

    def run(self):
        """Generator yielding the overall fraction for progress display."""
        self.create(np.arange(len(self.points)), IDENTITY)
        fine = int(np.clip(self.budget * 6, 1200, 24000))
        for _ in self.refine(fine, reframe=False, splits_only=True):
            yield .3 * min(1, self.total / fine)
        stop = self.budget * (.8 if self.chamfers else 1)
        start = max(self.total - stop, 1)
        for _ in self.merge(stop):
            yield .3 + .4 * min(1, 1 - (self.total - stop) / start)
        for _ in self.refine(self.budget, reframe=True):
            yield .7 + .3 * min(1, self.total / self.budget)

    def result(self):
        return list(self.blocks.values())


# --------------------------------------------------------------------------- output

def _inside(point, block, tolerance):
    return np.all(block.normals @ point <= block.offsets + tolerance)


def build_mesh(name, blocks, tolerance):
    """Join the convex blocks and drop faces that lie inside another block."""
    bm = bmesh.new()
    try:
        owner = []
        for number, block in enumerate(blocks):
            verts = [bm.verts.new(tuple(p)) for p in block.corners]
            result = bmesh.ops.convex_hull(bm, input=verts, use_existing_faces=False)
            unused = [v for v in result["geom_unused"] if isinstance(v, bmesh.types.BMVert)]
            interior = [v for v in result["geom_interior"] if isinstance(v, bmesh.types.BMVert)]
            for vert in unused + interior:
                if vert.is_valid and not vert.link_faces:
                    bm.verts.remove(vert)
            for face in result["geom"]:
                if isinstance(face, bmesh.types.BMFace):
                    face.tag = False
                    owner.append((face, number))
        lows = np.array([b.corners.min(axis=0) for b in blocks])
        highs = np.array([b.corners.max(axis=0) for b in blocks])
        hidden = []
        for face, number in owner:
            points = [np.array(v.co) for v in face.verts]
            points.append(sum(points) / len(points))
            low, high = np.min(points, axis=0), np.max(points, axis=0)
            nearby = np.nonzero(np.all(lows <= high + tolerance, axis=1)
                                & np.all(highs >= low - tolerance, axis=1))[0]
            for other in nearby:
                if other != number and all(_inside(p, blocks[other], tolerance) for p in points):
                    hidden.append(face)
                    break
        bmesh.ops.delete(bm, geom=hidden, context='FACES_ONLY')
        loose = [v for v in bm.verts if not v.link_faces]
        bmesh.ops.delete(bm, geom=loose, context='VERTS')
        bmesh.ops.recalc_face_normals(bm, faces=list(bm.faces))
        bmesh.ops.triangulate(bm, faces=list(bm.faces))
        data = bpy.data.meshes.new(name)
        bm.to_mesh(data)
        return data, len(hidden)
    finally:
        bm.free()


def material_color(material):
    if material:
        if material.use_nodes and material.node_tree:
            for node in material.node_tree.nodes:
                if node.type == 'BSDF_PRINCIPLED':
                    return tuple(node.inputs['Base Color'].default_value[:3])
        return tuple(material.diffuse_color[:3])
    return (.6, .6, .6)


def transfer_materials(result, sources):
    """Give each block face the material of the nearest source surface."""
    data = result.data
    slots, lookup, trees = [], {}, []
    for source in sources:
        mesh = source.data
        trees.append((BVHTree.FromPolygons([v.co for v in mesh.vertices],
                                           [p.vertices for p in mesh.polygons]), mesh))
    data.materials.clear()
    colors = []
    for face in data.polygons:
        best = None
        for tree, mesh in trees:
            hit = tree.find_nearest(face.center)
            if hit[0] is not None and (best is None or hit[3] < best[0]):
                best = (hit[3], mesh, hit[2])
        material = None
        if best:
            _, mesh, polygon = best
            slot = mesh.polygons[polygon].material_index
            material = mesh.materials[slot] if slot < len(mesh.materials) else None
        if material not in lookup:
            lookup[material] = len(slots)
            slots.append(material)
        face.material_index = lookup[material]
        colors.append(material_color(material))
    for material in slots:
        data.materials.append(material)
    return colors


def rebuild(sources, budget, resolution, chamfers):
    """Generator yielding progress; returns (mesh data, statistics)."""
    verts, tris = gather(sources)
    if not len(tris):
        raise ValueError("The selection contains no faces.")
    size = float(max(verts.max(axis=0) - verts.min(axis=0)))
    if size <= 0:
        raise ValueError("The selection has no extent.")
    points = sample_surface(verts, tris, size / resolution * .5)
    yield .1
    grid = Grid(points, resolution)
    yield .2
    builder = Rebuilder(points, grid, budget, chamfers)
    for fraction in builder.run():
        yield .2 + .7 * fraction
    blocks = builder.result()
    data, hidden = build_mesh("BrickWars Rebuild", blocks, size * 1e-5)
    error = sum(b.error for b in blocks)
    stats = {
        "blocks": len(blocks),
        "hidden_faces": hidden,
        "overfill": error / max(grid.solid_volume, 1),
        "voxel_size": grid.h,
        "reach": max(min(b.corners.max(axis=0) - b.corners.min(axis=0)) for b in blocks),
    }
    return data, stats
