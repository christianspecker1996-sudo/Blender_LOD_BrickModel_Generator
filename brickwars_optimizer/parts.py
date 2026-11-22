"""Rebuild every LEGO part as one simple convex shape; details go into the atlas.

Each part gets its own frame from its dominant face normals. Studs are trimmed
off, then a box, a polygonal cylinder, a rounded bar (liftarm) or a box with up
to two slope cuts is chosen by overlap with the part's convex hull. Faces that
cannot be seen from outside the assembly are removed afterwards.
"""
import heapq
import math

import bmesh
import bpy
import numpy as np
from mathutils import Vector
from mathutils.bvhtree import BVHTree

from . import instances
from .rebuild import material_color, polytope_vertices, triangle_count

AXIS_COS = math.cos(math.radians(6))
SLOPE_COS = math.cos(math.radians(12))
SAMPLES = 4096
IDENTITY = np.eye(3)


# --------------------------------------------------------------------------- analysis

def triangles_of(data):
    data.calc_loop_triangles()
    co = np.empty(len(data.vertices) * 3)
    data.vertices.foreach_get("co", co)
    co = co.reshape(-1, 3)
    tri = np.empty(len(data.loop_triangles) * 3, dtype=np.int64)
    data.loop_triangles.foreach_get("vertices", tri)
    tri = tri.reshape(-1, 3)
    a, b, c = co[tri[:, 0]], co[tri[:, 1]], co[tri[:, 2]]
    cross = np.cross(b - a, c - a)
    double_area = np.linalg.norm(cross, axis=1)
    keep = double_area > 1e-12
    normals = cross[keep] / double_area[keep, None]
    return co, normals, double_area[keep] / 2, ((a + b + c) / 3)[keep]


def dominant_direction(normals, areas, allowed):
    """Area-weighted direction shared by the most surface (sign ignored)."""
    if not allowed.any():
        return None
    candidates = np.nonzero(allowed)[0]
    candidates = candidates[np.argsort(-areas[candidates])[:48]]
    best, best_score = None, 0
    for i in candidates:
        aligned = allowed & (np.abs(normals @ normals[i]) > AXIS_COS)
        score = areas[aligned].sum()
        if score > best_score:
            signs = np.sign(normals[aligned] @ normals[i])[:, None]
            mean = (normals[aligned] * signs * areas[aligned, None]).sum(axis=0)
            best, best_score = mean / np.linalg.norm(mean), score
    return best


def part_frame(co, normals, areas):
    """Tightest of: dominant face normals, principal axes, model axes."""
    centered = co - co.mean(axis=0)
    _, vectors = np.linalg.eigh(centered.T @ centered)
    principal = vectors.T[::-1].copy()
    if np.linalg.det(principal) < 0:
        principal[2] *= -1
    frames = [normal_frame(co, normals, areas), principal, IDENTITY]
    volumes = [np.prod(np.ptp(co @ f.T, axis=0)) for f in frames]
    # Prefer the face-normal frame unless another one is clearly tighter.
    volumes[0] *= .97
    return frames[int(np.argmin(volumes))]


def normal_frame(co, normals, areas):
    first = dominant_direction(normals, areas, np.ones(len(normals), dtype=bool))
    if first is None:
        return IDENTITY
    second = dominant_direction(normals, areas, np.abs(normals @ first) < .1)
    if second is None:
        flat = co - co.mean(axis=0)
        flat -= np.outer(flat @ first, first)
        _, vectors = np.linalg.eigh(flat.T @ flat)
        second = vectors[:, -1]
    second = second - (second @ first) * first
    second /= np.linalg.norm(second)
    return np.array([first, second, np.cross(first, second)])


def trimmed_bounds(q, normals, areas, centers):
    """Box around the part body; studs and similar inset bumps are cut away."""
    low, high = q.min(axis=0), q.max(axis=0)
    extent = high - low
    bounds = [low.copy(), high.copy()]
    tolerance = 1e-3 * max(extent.max(), 1e-9)
    protrusion = 0.0
    for k in range(3):
        i, j = [a for a in range(3) if a != k]
        rect_area = max(extent[i] * extent[j], 1e-12)
        margin = .08 * min(extent[i], extent[j])
        for sign in (1, -1):
            facing = normals[:, k] * sign > AXIS_COS
            if not facing.any():
                continue
            levels = centers[facing, k] * sign
            level_area = areas[facing]
            extreme = high[k] if sign > 0 else -low[k]
            for level in np.unique(np.round(levels / tolerance))[::-1] * tolerance:
                if extreme - level > .5 * extent[k]:
                    break
                coverage = level_area[np.abs(levels - level) < 2 * tolerance].sum() / rect_area
                if coverage < .4:
                    continue
                beyond = q[q[:, k] * sign > level + 2 * tolerance]
                if not len(beyond):
                    break
                inset = (np.all(beyond[:, i] > low[i] + margin)
                         and np.all(beyond[:, i] < high[i] - margin)
                         and np.all(beyond[:, j] > low[j] + margin)
                         and np.all(beyond[:, j] < high[j] - margin))
                if inset:
                    if sign > 0:
                        bounds[1][k] = level
                    else:
                        bounds[0][k] = -level
                    protrusion = max(protrusion, extreme - level)
                break
    return bounds[0], bounds[1], protrusion


# --------------------------------------------------------------------------- candidates

def box_planes(low, high):
    normals = np.vstack((IDENTITY, -IDENTITY))
    return normals, np.concatenate((high, -low))


def prism_planes(polygon, k, low, high):
    """Planes of a convex 2D polygon (counter-clockwise in axes i, j) extruded along k."""
    i, j = [a for a in range(3) if a != k]
    normals, offsets = [], []
    for a, b in zip(polygon, np.roll(polygon, -1, axis=0)):
        edge = b - a
        normal2 = np.array([edge[1], -edge[0]])
        length = np.linalg.norm(normal2)
        if length < 1e-12:
            continue
        normal2 /= length
        normal = np.zeros(3)
        normal[i], normal[j] = normal2
        normals.append(normal)
        offsets.append(normal2 @ a)
    for sign in (1, -1):
        normal = np.zeros(3)
        normal[k] = sign
        normals.append(normal)
        offsets.append(high[k] if sign > 0 else -low[k])
    return np.array(normals), np.array(offsets)


def cylinder_polygon(center, radii, sides):
    # Area-preserving polygon: neither inscribed nor circumscribed.
    scale = math.sqrt(2 * math.pi / (sides * math.sin(2 * math.pi / sides)))
    angles = (np.arange(sides) + .5) * 2 * math.pi / sides
    return center + np.stack((np.cos(angles), np.sin(angles)), axis=1) * radii * scale


def frustum_planes(center, radii, sides, scales, k, low, high):
    """Polygonal cone stump along k with end radii scaled by scales (low, high)."""
    i, j = [a for a in range(3) if a != k]
    ring = cylinder_polygon(np.zeros(2), radii, sides)
    rings = []
    for scale, level in zip(scales, (low[k], high[k])):
        points = np.zeros((sides, 3))
        points[:, [i, j]] = center + ring * max(scale, .05)
        points[:, k] = level
        rings.append(points)
    middle = (rings[0].mean(axis=0) + rings[1].mean(axis=0)) / 2
    normals, offsets = [], []
    for m in range(sides):
        a, b, c = rings[0][m], rings[0][(m + 1) % sides], rings[1][m]
        normal = np.cross(b - a, c - a)
        normal /= np.linalg.norm(normal)
        if normal @ (middle - a) > 0:
            normal = -normal
        normals.append(normal)
        offsets.append(normal @ a)
    for sign in (1, -1):
        normal = np.zeros(3)
        normal[k] = sign
        normals.append(normal)
        offsets.append(high[k] if sign > 0 else -low[k])
    return np.array(normals), np.array(offsets)


def bar_polygon(low2, high2, segments):
    """Rounded bar along the longer 2D axis with half-round ends."""
    long_axis = int(np.argmax(high2 - low2))
    short = 1 - long_axis
    radius = (high2[short] - low2[short]) / 2
    middle = (high2[short] + low2[short]) / 2
    points = []
    for end, start_angle in ((high2[long_axis] - radius, -math.pi / 2),
                             (low2[long_axis] + radius, math.pi / 2)):
        for step in range(segments + 1):
            angle = start_angle + math.pi * step / segments
            r = radius if step in (0, segments) else radius / math.cos(math.pi / (2 * segments)) ** .5
            p = np.zeros(2)
            p[long_axis] = end + r * math.cos(angle)
            p[short] = middle + r * math.sin(angle)
            points.append(p)
    polygon = np.array(points)
    if long_axis == 1:
        polygon = polygon[::-1]
    return polygon


def hull_2d(points):
    """Counter-clockwise convex hull (monotone chain)."""
    points = np.unique(points, axis=0)
    if len(points) < 3:
        return points

    def half(sequence):
        chain = []
        for p in sequence:
            while len(chain) >= 2:
                a, b = chain[-2], chain[-1]
                if (b[0] - a[0]) * (p[1] - a[1]) - (b[1] - a[1]) * (p[0] - a[0]) > 0:
                    break
                chain.pop()
            chain.append(p)
        return chain

    lower = half(points)
    upper = half(points[::-1])
    return np.array(lower[:-1] + upper[:-1])


def polygon_area(polygon):
    x, y = polygon[:, 0], polygon[:, 1]
    return .5 * float(np.dot(x, np.roll(y, -1)) - np.dot(y, np.roll(x, -1)))


def profile_polygon(points, tolerance, max_corners):
    """Simplified outline: drop the corners that change the area least."""
    polygon = hull_2d(points)
    if len(polygon) < 3:
        return None
    full = polygon_area(polygon)
    if full <= 0:
        return None
    while len(polygon) > 3:
        prev, nxt = np.roll(polygon, 1, axis=0), np.roll(polygon, -1, axis=0)
        loss = .5 * np.abs((polygon[:, 0] - prev[:, 0]) * (nxt[:, 1] - prev[:, 1])
                           - (nxt[:, 0] - prev[:, 0]) * (polygon[:, 1] - prev[:, 1]))
        weakest = int(np.argmin(loss))
        trial = np.delete(polygon, weakest, axis=0)
        if len(polygon) <= max_corners and (full - polygon_area(trial)) / full > tolerance:
            break
        polygon = trial
    # Grow back to the original area so the outline neither shrinks nor bloats.
    center = polygon.mean(axis=0)
    return center + (polygon - center) * math.sqrt(full / polygon_area(polygon))


def slope_planes(q, normals_q, areas, centers_q, low, high):
    """Up to two large non-axis face directions of the part, as outermost planes."""
    oblique = np.max(np.abs(normals_q), axis=1) < SLOPE_COS
    extent = high - low
    surface = 2 * float(extent @ np.roll(extent, 1))
    found = []
    remaining = oblique.copy()
    for _ in range(2):
        direction = dominant_direction(normals_q, areas, remaining)
        if direction is None:
            break
        # dominant_direction ignores sign; orient along the outward normals.
        group = remaining & (normals_q @ direction > AXIS_COS)
        flipped = remaining & (normals_q @ -direction > AXIS_COS)
        if areas[flipped].sum() > areas[group].sum():
            direction, group = -direction, flipped
        remaining &= np.abs(normals_q @ direction) <= AXIS_COS
        if areas[group].sum() < .05 * surface:
            continue
        found.append((direction, float((centers_q[group] @ direction).max())))
    return found


def revolve_points(q, k, center, radii, low, high, rings, sides):
    """Body of revolution along k following the part's radial profile.

    Dishes and domes are curved; a cone stump with two rings misses them badly.
    The half-profile (height, radius) is simplified to `rings` rings plus the axis.
    """
    i, j = [a for a in range(3) if a != k]
    inside = (q[:, k] >= low[k]) & (q[:, k] <= high[k])
    radial = np.sqrt((((q[inside][:, [i, j]] - center) / radii) ** 2).sum(axis=1))
    height = q[inside, k]
    if not len(height):
        return None
    step = (high[k] - low[k]) / 64 or 1e-9
    points = np.column_stack((np.round(height / step) * step, np.minimum(radial, 1.0)))
    points = np.vstack((points, [[low[k], 0.0], [high[k], 0.0]]))
    polygon = hull_2d(np.unique(np.round(points, 4), axis=0))
    if len(polygon) < 3:
        return None
    # Drop the corners that change the profile area least until `rings` remain off-axis.
    while True:
        off_axis = polygon[:, 1] > .02
        if np.count_nonzero(off_axis) <= rings or len(polygon) <= 3:
            break
        prev, nxt = np.roll(polygon, 1, axis=0), np.roll(polygon, -1, axis=0)
        loss = .5 * np.abs((polygon[:, 0] - prev[:, 0]) * (nxt[:, 1] - prev[:, 1])
                           - (nxt[:, 0] - prev[:, 0]) * (polygon[:, 1] - prev[:, 1]))
        loss[~off_axis] = np.inf
        polygon = np.delete(polygon, int(np.argmin(loss)), axis=0)
    result = []
    ring_levels = polygon[polygon[:, 1] > .02, 0]
    for level, radius in polygon:
        if radius <= .02:
            # An axis point level with a ring lies inside that end cap.
            if np.any(np.abs(ring_levels - level) < step / 2):
                continue
            p = np.zeros(3)
            p[[i, j]], p[k] = center, level
            result.append(p[None])
            continue
        ring = np.zeros((sides, 3))
        ring[:, [i, j]] = cylinder_polygon(center, radii * radius, sides)
        ring[:, k] = level
        result.append(ring)
    return np.vstack(result)


CYLINDER_SIDES = (6, 8, 10, 12, 16, 20)
DISH_RINGS = (3, 4)
ROUND_KINDS = {'cylinder', 'cone', 'dish', 'bar', 'profile'}
ROUND_BONUS = 1.5
FACET_WEIGHT = 1.5
BAR_SEGMENTS = (1, 2, 3, 4, 6)
PROFILE_STEPS = ((.06, 6), (.03, 10), (.015, 16), (.007, 24))


def candidates(q, normals_q, areas, centers_q, low, high):
    """Every shape at several levels of roundness; the budget picks among them."""
    box = box_planes(low, high)
    yield 'box', box, 0
    for k in range(3):
        i, j = [a for a in range(3) if a != k]
        low2, high2 = low[[i, j]], high[[i, j]]
        size2 = high2 - low2
        if min(size2) <= 0:
            continue
        ratio = max(size2) / min(size2)
        center = (low2 + high2) / 2
        if ratio < 1.35:
            for sides in CYLINDER_SIDES:
                yield 'cylinder', prism_planes(cylinder_polygon(center, size2 / 2, sides),
                                               k, low, high), sides
            # Cones and dishes: radius measured separately near both ends.
            radial = np.sqrt((((q[:, [i, j]] - center) / (size2 / 2)) ** 2).sum(axis=1))
            t = (q[:, k] - low[k]) / max(high[k] - low[k], 1e-12)
            ends = [radial[t <= .15], radial[t >= .85]]
            if all(len(e) for e in ends):
                scales = [min(1.0, float(e.max())) for e in ends]
                if min(scales) < .85:
                    for sides in CYLINDER_SIDES:
                        yield 'cone', frustum_planes(center, size2 / 2, sides, scales,
                                                     k, low, high), sides
            # Curved dishes and domes: several rings along the real profile.
            for rings in DISH_RINGS:
                for sides in CYLINDER_SIDES:
                    points = revolve_points(q, k, center, size2 / 2, low, high, rings, sides)
                    if points is not None and len(points) >= 4:
                        yield 'dish', hull_planes(points) + (points,), sides
        else:
            for segments in BAR_SEGMENTS:
                yield 'bar', prism_planes(bar_polygon(low2, high2, segments), k, low, high), 2 * segments
        # Outline along this axis, e.g. the arc of a curved slope seen from the side.
        inside = (q[:, k] >= low[k]) & (q[:, k] <= high[k])
        flat = np.clip(q[inside][:, [i, j]], low2, high2)
        if len(flat) >= 3:
            step = max(size2) / 256
            flat = np.unique(np.round(flat / step), axis=0) * step
            for tolerance, corners in PROFILE_STEPS:
                polygon = profile_polygon(flat, tolerance, corners)
                if polygon is not None and len(polygon) > 4:
                    yield 'profile', prism_planes(polygon, k, low, high), 0
    slopes = slope_planes(q, normals_q, areas, centers_q, low, high)
    for chosen in slopes:
        yield 'slope', (np.vstack((box[0], chosen[0])), np.append(box[1], chosen[1])), 0
    if len(slopes) == 2:
        yield 'slope', (np.vstack((box[0], slopes[0][0], slopes[1][0])),
                        np.append(box[1], [slopes[0][1], slopes[1][1]])), 0


def hull_planes(q):
    bm = bmesh.new()
    try:
        verts = [bm.verts.new(tuple(p)) for p in q]
        bmesh.ops.convex_hull(bm, input=verts, use_existing_faces=False)
        bm.normal_update()
        normals = np.array([tuple(f.normal) for f in bm.faces])
        offsets = np.array([f.normal.dot(f.verts[0].co) for f in bm.faces])
        # Orient every plane so the hull's centroid lies inside.
        inward = normals @ q.mean(axis=0) > offsets
        normals[inward] *= -1
        offsets[inward] *= -1
        return normals, offsets
    finally:
        bm.free()


class Option:
    __slots__ = ("kind", "normals", "offsets", "corners", "tris", "iou", "quality")


class Fit:
    """A part with its Pareto-optimal shape options, cheapest first."""
    __slots__ = ("name", "options", "choice", "weight", "factor", "color", "protrusion", "source",
                 "group", "transform", "template")

    @property
    def option(self):
        return self.options[self.choice]

    kind = property(lambda self: self.option.kind)
    tris = property(lambda self: self.option.tris)
    iou = property(lambda self: self.option.iou)
    normals = property(lambda self: self.option.normals)
    offsets = property(lambda self: self.option.offsets)
    corners = property(lambda self: self.option.corners)


def fit_part(source, rng):
    co, normals, areas, centers = triangles_of(source.data)
    if not len(areas):
        return None
    frame = part_frame(co, normals, areas)
    q, normals_q, centers_q = co @ frame.T, normals @ frame.T, centers @ frame.T
    low, high, protrusion = trimmed_bounds(q, normals_q, areas, centers_q)
    extent = high - low
    floor = 1e-4 * max(extent.max(), 1e-9)
    grow = np.maximum(0, floor - extent) / 2
    low, high = low - grow, high + grow
    extent = high - low
    if len(q) > 6000:
        q_hull = q[np.unique(np.round((q - low) / (extent.max() / 200)), axis=0,
                             return_index=True)[1]]
    else:
        q_hull = q
    ref_normals, ref_offsets = hull_planes(q_hull)
    tolerance = 1e-6 * max(extent.max(), 1e-9)
    samples = low + rng.random((SAMPLES, 3)) * extent
    reference = np.all(samples @ ref_normals.T <= ref_offsets + tolerance, axis=1)
    options = []
    for kind, shape, sides in candidates(q, normals_q, areas, centers_q, low, high):
        cand_normals, cand_offsets = shape[:2]
        # Shapes built from points keep them as corners; planes of a triangulated
        # hull would otherwise yield near-duplicate vertices.
        corners = shape[2] if len(shape) > 2 else polytope_vertices(
            cand_normals, cand_offsets, tolerance * 10)
        if len(corners) < 4:
            continue
        inside = np.all(samples @ cand_normals.T <= cand_offsets + tolerance, axis=1)
        union = np.count_nonzero(reference | inside)
        option = Option()
        option.kind, option.tris = kind, triangle_count(len(corners))
        option.iou = np.count_nonzero(reference & inside) / union if union else 0
        # Facets of a polygonal circle are visible long before they cost area.
        option.quality = option.iou - (FACET_WEIGHT * (1 - math.cos(math.pi / sides))
                                       if sides else 0)
        option.normals, option.offsets = cand_normals @ frame, cand_offsets
        option.corners = corners @ frame
        options.append(option)
    # Clearly round parts (dishes, barrels, wheels) never fall back to a square:
    # angular options that fit worse than the round ones are dropped.
    revolved = [o.iou for o in options if o.kind in ('cylinder', 'cone', 'dish')]
    boxes = [o.iou for o in options if o.kind == 'box']
    if revolved and boxes and max(revolved) >= max(.6, boxes[0] + .15):
        floor = max(revolved) - .05
        options = [o for o in options
                   if o.kind in ('cylinder', 'cone', 'dish') or o.iou >= floor]
    # Keep only options that are better than every cheaper one.
    options.sort(key=lambda o: (o.tris, -o.quality))
    # Option 0 leaves the part out (texture only) when the budget is too small.
    omit = Option()
    omit.kind, omit.tris, omit.iou, omit.quality = 'omit', 0, 0.0, 0.0
    omit.normals, omit.offsets, omit.corners = None, None, np.empty((0, 3))
    pareto = [omit]
    for option in options:
        if option.quality > pareto[-1].quality + .003:
            pareto.append(option)
    if len(pareto) < 2:
        return None
    fit = Fit()
    fit.name, fit.options, fit.choice, fit.factor = source.name, pareto, 1, 1.0
    # Visual weight: surface of the part's box, so big parts get detail first.
    fit.weight = 2 * float(extent @ np.roll(extent, 1))
    fit.protrusion = protrusion
    fit.source = source
    material = source.data.materials[0] if source.data.materials else None
    fit.color = material_color(material)
    fit.group, fit.transform, fit.template = None, instances.IDENTITY, None
    return fit


def moved(option, rotation, offset):
    """The option of a leader placed on an identical part (leader = R @ part + t)."""
    copy = Option()
    copy.kind, copy.tris, copy.iou, copy.quality = option.kind, option.tris, option.iou, option.quality
    copy.corners = (option.corners - offset) @ rotation
    if option.normals is None:
        copy.normals, copy.offsets = None, None
    else:
        copy.normals = option.normals @ rotation
        copy.offsets = option.offsets - option.normals @ offset
    return copy


def placed(leader, source, transform):
    """Fit for a part identical to leader: same shape options, moved into place."""
    fit = Fit()
    rotation, offset = transform
    fit.name, fit.source, fit.transform, fit.template = source.name, source, transform, None
    fit.options = [moved(o, rotation, offset) for o in leader.options]
    fit.choice, fit.factor, fit.weight = leader.choice, 1.0, leader.weight
    fit.protrusion, fit.color = leader.protrusion, leader.color
    return fit


def members_of(fits):
    groups = {}
    for number, fit in enumerate(fits):
        groups.setdefault(fit.group, []).append(number)
    return groups


def allocate(fits, budget):
    """Greedy: spend triangles on the upgrade with the best weighted gain per triangle.

    Identical parts share one texture region, so they always take the same shape.
    """
    # Costs are scaled by each part's visible share, measured after hidden-face removal.
    groups = members_of(fits)
    for fit in fits:
        fit.choice = 0
    total = 0.0
    heap = []

    def offer(group):
        members = groups[group]
        lead = fits[group]
        current = lead.option
        weight = sum(fits[m].weight for m in members)
        best = None
        for choice in range(lead.choice + 1, len(lead.options)):
            option = lead.options[choice]
            cost = sum((option.tris - current.tris) * fits[m].factor for m in members)
            if cost <= 0:
                continue
            # Faceted round parts stand out most, so roundness gets a bonus.
            bonus = ROUND_BONUS if option.kind in ROUND_KINDS else 1.0
            gain = weight * bonus * (option.quality - current.quality) / cost
            if best is None or gain > best[0]:
                best = (gain, choice)
        if best and best[0] > 0:
            heapq.heappush(heap, (-best[0], group, lead.choice, best[1]))

    for group in groups:
        offer(group)
    while heap:
        _, group, previous, choice = heapq.heappop(heap)
        lead = fits[group]
        if lead.choice != previous:
            continue
        added = sum((lead.options[choice].tris - lead.tris) * fits[m].factor
                    for m in groups[group])
        if total + added > budget:
            continue
        for m in groups[group]:
            fits[m].choice = choice
        total += added
        offer(group)
    return total


# --------------------------------------------------------------------------- visibility

def occluder(sources):
    """One BVH over all sources plus a lookup from triangle to part number."""
    verts, polys, owner = [], [], []
    for number, source in enumerate(sources):
        data = source.data
        co = np.empty(len(data.vertices) * 3)
        data.vertices.foreach_get("co", co)
        base = len(verts)
        verts.extend(map(tuple, co.reshape(-1, 3)))
        for tri in data.loop_triangles:
            polys.append([base + v for v in tri.vertices])
            owner.append(number)
    return BVHTree.FromPolygons(verts, polys, all_triangles=True), owner


def hemisphere(normal):
    normal = Vector(normal)
    tangent = normal.orthogonal().normalized()
    bitangent = normal.cross(tangent)
    directions = [normal]
    for step in range(8):
        angle = step * math.pi / 4
        side = tangent * math.cos(angle) + bitangent * math.sin(angle)
        directions.append((normal * .65 + side * .76).normalized())
    return directions


def face_visible(tree, owner, part, points, normal, offset, reach):
    directions = hemisphere(normal)
    for point in points:
        start = Vector(point) + Vector(normal) * offset
        for direction in directions:
            origin = start
            for _ in range(6):
                hit, _, index, distance = tree.ray_cast(origin, direction, reach)
                if hit is None:
                    return True
                if owner[index] != part:
                    break
                origin = hit + direction * offset * .1
            else:
                return True
    return False


# --------------------------------------------------------------------------- output

def template(corners):
    """Triangulated convex hull as (vertices, triangles); copies keep its face order."""
    bm = bmesh.new()
    try:
        verts = [bm.verts.new(tuple(p)) for p in corners]
        bmesh.ops.convex_hull(bm, input=verts, use_existing_faces=False)
        for vert in list(bm.verts):
            if not vert.link_faces:
                bm.verts.remove(vert)
        bmesh.ops.recalc_face_normals(bm, faces=list(bm.faces))
        bmesh.ops.triangulate(bm, faces=list(bm.faces))
        bm.verts.index_update()
        co = np.array([tuple(v.co) for v in bm.verts])
        tris = np.array([[v.index for v in f.verts] for f in bm.faces], dtype=np.int64)
        return co, tris
    finally:
        bm.free()


def build(fits, sources, cull, size):
    """One joined mesh with face attributes for the part number and template face."""
    tree, owner = occluder(sources) if cull else (None, None)
    offset = 2e-3 * size
    bm = bmesh.new()
    layer = bm.faces.layers.int.new("brickwars_part")
    face_layer = bm.faces.layers.int.new("brickwars_face")
    hidden_faces = 0
    for number, fit in enumerate(fits):
        if fit.group == number:
            fit.template = None if fit.kind == 'omit' else template(fit.corners)
    try:
        for number, fit in enumerate(fits):
            if fit.kind == 'omit':
                continue
            co, tris = fits[fit.group].template
            rotation, shift = fit.transform
            verts = [bm.verts.new(tuple(p)) for p in (co - shift) @ rotation]
            faces = []
            for local, tri in enumerate(tris):
                face = bm.faces.new([verts[i] for i in tri])
                face[layer], face[face_layer] = number, local
                face.normal_update()
                faces.append(face)
            if not cull:
                continue
            hidden = []
            for face in faces:
                corners = [v.co.copy() for v in face.verts]
                center = sum(corners, Vector()) / len(corners)
                points = [center] + [center.lerp(c, .7) for c in corners]
                if not face_visible(tree, owner, number, points, face.normal, offset, size * 4):
                    hidden.append(face)
            if hidden:
                hidden_faces += len(hidden)
                bmesh.ops.delete(bm, geom=hidden, context='FACES')
        bmesh.ops.triangulate(bm, faces=list(bm.faces))
        data = bpy.data.meshes.new("BrickWars Parts")
        bm.to_mesh(data)
        return data, hidden_faces
    finally:
        bm.free()


def split(joined, fits, collection, originals):
    """Separate the joined mesh into one object per visible part."""
    parts = np.empty(len(joined.data.polygons), dtype=np.int64)
    joined.data.attributes["brickwars_part"].data.foreach_get("value", parts)
    objects = []
    for number in np.unique(parts):
        bm = bmesh.new()
        try:
            bm.from_mesh(joined.data)
            layer = bm.faces.layers.int["brickwars_part"]
            bmesh.ops.delete(bm, geom=[f for f in bm.faces if f[layer] != number], context='FACES')
            data = bpy.data.meshes.new(fits[number].name + "_LP")
            bm.to_mesh(data)
        finally:
            bm.free()
        for name in ("brickwars_part", "brickwars_face"):
            if name in data.attributes:
                data.attributes.remove(data.attributes[name])
        # Each part only carries the materials it uses (atlas, its stud colors).
        indices = np.empty(len(data.polygons), dtype=np.int64)
        data.polygons.foreach_get("material_index", indices)
        used = sorted(set(indices.tolist())) or [0]
        for old in used:
            if old < len(joined.data.materials):
                data.materials.append(joined.data.materials[old])
        lookup = np.zeros(max(used) + 1, dtype=np.int64)
        lookup[used] = np.arange(len(used))
        data.polygons.foreach_set("material_index", lookup[indices])
        original = originals[number]
        world = joined.matrix_world.copy()
        data.transform(original.matrix_world.inverted() @ world)
        obj = bpy.data.objects.new(original.name + "_LP", data)
        obj.matrix_world = original.matrix_world
        collection.objects.link(obj)
        objects.append(obj)
    return objects


def rebuild_parts(sources, budget, cull, share=False, upright=True, occluders=None):
    """Generator yielding progress; returns (joined mesh data, fits, kept indices, stats).

    With share, identical parts are fitted once and become copies of their leader,
    so they can share one texture region (see atlas_layout).
    """
    rng = np.random.default_rng(7)
    task = instances.classify(sources, share, upright)
    while True:
        try:
            yield .1 * next(task)
        except StopIteration as finished:
            leaders, transforms = finished.value
            break
    fits = [None] * len(sources)
    for number, source in enumerate(sources):
        if leaders[number] == number:
            fits[number] = fit_part(source, rng)
            if number % 8 == 0:
                yield .1 + .5 * number / len(sources)
    for number, source in enumerate(sources):
        lead = fits[leaders[number]]
        if leaders[number] != number and lead is not None:
            fits[number] = placed(lead, source, transforms[number])
    keep = [i for i, fit in enumerate(fits) if fit is not None]
    index = {old: new for new, old in enumerate(keep)}
    for old in keep:
        fits[old].group = index[leaders[old]]
    fits = [fits[i] for i in keep]
    used = [sources[i] for i in keep]
    # Visibility may be tested against other meshes (the parts with their studs).
    blockers = [occluders[i] for i in keep] if occluders else used
    corners = np.vstack([f.options[1].corners for f in fits])
    size = float(max(corners.max(axis=0) - corners.min(axis=0)))
    # Pass 1 builds every part as its cheapest shape to see what is visible.
    for fit in fits:
        fit.choice = 1
    data, hidden_faces = build(fits, blockers, cull, size)
    yield .7
    if cull:
        numbers = np.empty(len(data.polygons), dtype=np.int64)
        data.attributes["brickwars_part"].data.foreach_get("value", numbers)
        area = np.empty(len(data.polygons))
        data.polygons.foreach_get("area", area)
        shown = np.bincount(numbers, minlength=len(fits))
        shown_area = np.bincount(numbers, weights=area, minlength=len(fits))
        for fit, count, part_area in zip(fits, shown, shown_area):
            fit.factor = count / max(fit.tris, 1)
            fit.weight = min(fit.weight, float(part_area) * 1.5)
    # Later passes spend the budget and correct for the real count after culling.
    spend = budget
    for attempt in range(4):
        allocate(fits, spend)
        bpy.data.meshes.remove(data)
        data, hidden_faces = build(fits, blockers, cull, size)
        yield .75 + .05 * attempt
        after = sum(len(p.vertices) - 2 for p in data.polygons)
        if after > budget:
            spend -= (after - budget) * 1.1
        elif after < .96 * budget and spend < budget * 3:
            spend += (budget - after) * .9
        else:
            break
    while after > budget and spend > 0:
        spend -= (after - budget) * 1.1 + 1
        allocate(fits, spend)
        bpy.data.meshes.remove(data)
        data, hidden_faces = build(fits, blockers, cull, size)
        after = sum(len(p.vertices) - 2 for p in data.polygons)
    numbers = np.empty(len(data.polygons), dtype=np.int64)
    data.attributes["brickwars_part"].data.foreach_get("value", numbers)
    visible = set(numbers.tolist())
    kinds = {}
    for fit in fits:
        kinds[fit.kind] = kinds.get(fit.kind, 0) + 1
    stats = {
        "parts": len(fits),
        "omitted_parts": sum(f.kind == 'omit' for f in fits),
        "hidden_parts": len(fits) - len(visible),
        "hidden_faces": hidden_faces,
        "kinds": kinds,
        "protrusion": max((f.protrusion for f in fits), default=0.0),
        "size": size,
        "mean_iou": float(np.mean([f.iou for f in fits])),
        "part_types": len({f.group for f in fits}),
    }
    return data, fits, keep, stats


# --------------------------------------------------------------------------- shared atlas

def face_keys(data, fits):
    """Group and template face of every polygon in the joined mesh."""
    count = len(data.polygons)
    numbers = np.empty(count, dtype=np.int64)
    local = np.empty(count, dtype=np.int64)
    data.attributes["brickwars_part"].data.foreach_get("value", numbers)
    data.attributes["brickwars_face"].data.foreach_get("value", local)
    groups = np.array([f.group for f in fits], dtype=np.int64)[numbers]
    return numbers, groups, local


def atlas_layout(data, fits, size, reach):
    """Mesh that owns the atlas, with one copy of every texture region.

    A part type seen more than once is laid out once, beside the model, with its
    leader's source, so the bake sees that part alone. Unique parts stay in place
    and are baked in context. Returns (mesh data, fallback colors, sources to move).
    """
    numbers, groups, local = face_keys(data, fits)
    visible = {}
    for number, group in zip(numbers.tolist(), groups.tolist()):
        visible.setdefault(group, set()).add(number)
    used = {}
    for group, face in zip(groups.tolist(), local.tolist()):
        used.setdefault(group, set()).add(face)
    shared = sorted(g for g, members in visible.items() if len(members) > 1)
    co = np.empty(len(data.vertices) * 3)
    data.vertices.foreach_get("co", co)
    co = co.reshape(-1, 3)
    high = co.max(axis=0)
    cell = max((float(np.ptp(fits[g].template[0], axis=0).max()) for g in shared), default=0)
    spacing = cell + 4 * reach + .1 * size
    columns = max(1, math.ceil(math.sqrt(len(shared))))
    shifts = {}
    for slot, group in enumerate(shared):
        row, column = divmod(slot, columns)
        corner = np.array([high[0] + spacing * (column + 1), spacing * row, 0.0])
        shifts[group] = corner - fits[group].template[0].min(axis=0)
    # Where a part baked alone shows nothing (through-holes, empty hull corners),
    # its most visible copy in the model fills in what lies behind.
    shown = np.bincount(numbers, minlength=len(fits))
    bm = bmesh.new()
    key_layer = bm.faces.layers.int.new("brickwars_key")
    context_layers = [bm.verts.layers.float.new("brickwars_context_" + a) for a in "xyz"]
    fallback = []
    try:
        for group in sorted(used):
            co_t, tris = fits[group].template
            number = max(visible[group], key=lambda n: shown[n])
            rotation, shift = fits[number].transform
            in_place, color = (co_t - shift) @ rotation, fits[number].color
            points = co_t + shifts[group] if group in shifts else in_place
            verts = [bm.verts.new(tuple(p)) for p in points]
            for vert, place in zip(verts, in_place):
                for layer, value in zip(context_layers, place):
                    vert[layer] = value
            for face in sorted(used[group]):
                created = bm.faces.new([verts[i] for i in tris[face]])
                created[key_layer] = group * 4096 + face
                fallback.append(color)
            for vert in verts:
                if not vert.link_faces:
                    bm.verts.remove(vert)
        layout = bpy.data.meshes.new("BrickWars Atlas Layout")
        bm.to_mesh(layout)
    finally:
        bm.free()
    moves = [(fits[g].source, shifts[g]) for g in shared]
    shared_parts = sum(len(visible[g]) for g in shared)
    return layout, fallback, moves, {"shared_parts": shared_parts, "texture_groups": len(shared)}


def in_context(layout):
    """Copy of the layout with every shared part moved back onto a copy in the model."""
    data = layout.copy()
    co = np.empty((len(data.vertices), 3))
    for axis, name in enumerate("xyz"):
        values = np.empty(len(data.vertices))
        data.attributes["brickwars_context_" + name].data.foreach_get("value", values)
        co[:, axis] = values
    data.vertices.foreach_set("co", co.ravel())
    data.update()
    return data


def copy_uvs(layout, data, fits):
    """Give every polygon the UVs of its texture region in the layout mesh."""
    keys = np.empty(len(layout.polygons), dtype=np.int64)
    layout.attributes["brickwars_key"].data.foreach_get("value", keys)
    source_uv = np.empty(len(layout.loops) * 2)
    layout.uv_layers.active.uv.foreach_get("vector", source_uv)
    starts = np.empty(len(layout.polygons), dtype=np.int64)
    layout.polygons.foreach_get("loop_start", starts)
    source_uv = source_uv.reshape(-1, 2)
    lookup = {k: source_uv[s:s + 3] for k, s in zip(keys.tolist(), starts.tolist())}
    _, groups, local = face_keys(data, fits)
    for layer in list(data.uv_layers):
        data.uv_layers.remove(layer)
    target = data.uv_layers.new(name="BrickWars UV")
    uv = np.empty((len(data.loops), 2))
    targets = np.empty(len(data.polygons), dtype=np.int64)
    data.polygons.foreach_get("loop_start", targets)
    for start, key in zip(targets.tolist(), (groups * 4096 + local).tolist()):
        uv[start:start + 3] = lookup[key]
    target.uv.foreach_set("vector", uv.ravel())
