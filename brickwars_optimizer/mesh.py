"""Evaluated mesh snapshots and topology-aware simplification."""
import math
from collections import defaultdict

import bmesh
import bpy
from mathutils.bvhtree import BVHTree


def triangles(mesh):
    return sum(len(face.vertices) - 2 for face in mesh.polygons)


def analyze(objects):
    meshes = [obj.data for obj in objects]
    return {
        "objects": len(meshes),
        "faces": sum(len(mesh.polygons) for mesh in meshes),
        "triangles": sum(map(triangles, meshes)),
        "materials": len({mat for mesh in meshes for mat in mesh.materials if mat}),
    }


def activate(context, obj, others=()):
    for selected in list(context.selected_objects):
        selected.select_set(False)
    for selected in (*others, obj):
        selected.select_set(True)
    context.view_layer.objects.active = obj


def snapshot(context, obj, collection, origin):
    mesh = bpy.data.meshes.new_from_object(
        obj.evaluated_get(context.evaluated_depsgraph_get()),
        preserve_all_data_layers=True, depsgraph=context.evaluated_depsgraph_get(),
    )
    transform = obj.matrix_world.copy()
    transform.translation -= origin
    mesh.transform(transform)
    if transform.is_negative:
        mesh.flip_normals()
    copy = bpy.data.objects.new(obj.name + " Source", mesh)
    collection.objects.link(copy)
    copy.location = origin
    return copy


def remove_object(obj):
    mesh = obj.data
    bpy.data.objects.remove(obj, do_unlink=True)
    if mesh.users == 0:
        bpy.data.meshes.remove(mesh)


def planar(mesh):
    bm = bmesh.new()
    try:
        bm.from_mesh(mesh)
        bmesh.ops.dissolve_limit(
            bm, angle_limit=math.radians(0.35), use_dissolve_boundaries=False,
            verts=list(bm.verts), edges=list(bm.edges),
            delimit={'NORMAL', 'MATERIAL', 'SHARP', 'SEAM'},
        )
        bm.to_mesh(mesh)
    finally:
        bm.free()


def stitch_import_seams(mesh):
    """Join unambiguous, oppositely wound splits within one imported object."""
    bm = bmesh.new()
    try:
        bm.from_mesh(mesh)
        bm.normal_update()
        boundary = defaultdict(list)
        for edge in bm.edges:
            if not edge.is_boundary:
                continue
            loop = edge.link_loops[0]
            # Use exact positions so nearby LEGO pieces are never snapped together.
            start = tuple(loop.vert.co)
            end = tuple(loop.link_loop_next.vert.co)
            if start != end:
                boundary[tuple(sorted((start, end)))].append((start, end, loop))
        targets = {}
        hard_positions = set()
        for group in boundary.values():
            if len(group) != 2:
                continue
            (start, end, a), (other_start, other_end, b) = group
            if start != other_end or end != other_start or a.face.normal.dot(b.face.normal) < -.95:
                continue
            if a.face.normal.dot(b.face.normal) < math.cos(math.radians(45)):
                hard_positions.add(tuple(sorted((start, end))))
            for va, vb in ((a.vert, b.link_loop_next.vert), (a.link_loop_next.vert, b.vert)):
                while va in targets:
                    va = targets[va]
                while vb in targets:
                    vb = targets[vb]
                if va != vb:
                    targets[vb] = va
        if not targets:
            return 0
        for vert, target in targets.items():
            while target in targets:
                target = targets[target]
            targets[vert] = target
        faces = len(bm.faces)
        non_manifold = sum(len(e.link_faces) > 2 for e in bm.edges)
        bmesh.ops.weld_verts(bm, targetmap=targets)
        if len(bm.faces) != faces or sum(len(e.link_faces) > 2 for e in bm.edges) > non_manifold:
            return 0
        bm.to_mesh(mesh)
        return len(targets)
    finally:
        bm.free()


def clean(mesh):
    """Remove same-facing duplicate faces within a snapshot, preserving seams."""
    bm = bmesh.new()
    try:
        bm.from_mesh(mesh)
        bm.normal_update()
        uv_layers = list(bm.loops.layers.uv.values())
        seen = set()
        duplicates = []
        for face in bm.faces:
            key = (tuple(sorted((tuple(loop.vert.co), tuple(tuple(loop[layer].uv) for layer in uv_layers))
                                for loop in face.loops)),
                   tuple(round(n, 6) for n in face.normal), face.material_index)
            if key in seen:
                duplicates.append(face)
            seen.add(key)
        for face in duplicates:
            bm.faces.remove(face)
        for edge in list(bm.edges):
            if not edge.link_faces:
                bm.edges.remove(edge)
        for vert in list(bm.verts):
            if not vert.link_edges:
                bm.verts.remove(vert)
        bm.to_mesh(mesh)
        return len(duplicates)
    finally:
        bm.free()


def topology(mesh):
    counts = defaultdict(int)
    for face in mesh.polygons:
        for edge in face.edge_keys:
            counts[tuple(sorted(edge))] += 1
    return sum(n == 1 for n in counts.values()), sum(n > 2 for n in counts.values())


def remove_enclosed_boxes(mesh):
    """Remove closed components strictly inside a verified closed solid box."""
    bm = bmesh.new()
    try:
        bm.from_mesh(mesh)
        unseen = set(bm.faces)
        components = []
        boxes = []
        while unseen:
            seed = unseen.pop()
            faces = [seed]
            for face in faces:
                for edge in face.edges:
                    for neighbor in edge.link_faces:
                        if neighbor in unseen:
                            unseen.remove(neighbor)
                            faces.append(neighbor)
            if any(not e.is_manifold or not e.is_contiguous for f in faces for e in f.edges):
                continue
            verts = {v for f in faces for v in f.verts}
            axes = [sorted({v.co[i] for v in verts}) for i in range(3)]
            low, high = zip(*[(axis[0], axis[-1]) for axis in axes])
            bounds = (low, high)
            components.append((bounds, faces))
            if len(faces) != 6 or len(verts) != 8 or any(len(axis) != 2 for axis in axes):
                continue
            if any(len(f.verts) != 4 for f in faces):
                continue
            planes = {(i, f.verts[0].co[i]) for f in faces for i in range(3)
                      if all(v.co[i] == f.verts[0].co[i] for v in f.verts)}
            if len(planes) == 6:
                boxes.append(bounds)
        removed = []
        for (low, high), faces in components:
            for outer_low, outer_high in boxes:
                epsilon = max(b-a for a, b in zip(outer_low, outer_high)) * 1e-5
                if all(a+epsilon < x <= y < b-epsilon
                       for a, x, y, b in zip(outer_low, low, high, outer_high)):
                    removed.extend(faces)
                    break
        if removed:
            bmesh.ops.delete(bm, geom=removed, context='FACES')
            bm.to_mesh(mesh)
        return len(removed)
    finally:
        bm.free()


def surface_tree(mesh):
    return BVHTree.FromPolygons([v.co for v in mesh.vertices],
                                [p.vertices for p in mesh.polygons])


def close_to_surface(mesh, tree, tolerance):
    stride = max(1, len(mesh.vertices) // 3500)
    return all(tree.find_nearest(mesh.vertices[i].co, tolerance)[0] is not None
               for i in range(0, len(mesh.vertices), stride))


def collapse(context, obj, target):
    count = triangles(obj.data)
    if count <= target:
        return
    baseline = topology(obj.data)
    original_tree = surface_tree(obj.data)
    tolerance = max(1e-7, max(obj.dimensions) * .012)
    lower = ratio = max(1 / count, target / count)
    upper = 1.0
    best = None
    try:
        for _ in range(7):
            modifier = obj.modifiers.new("BrickWars Collapse", 'DECIMATE')
            modifier.ratio = ratio
            modifier.use_collapse_triangulate = True
            try:
                candidate = bpy.data.meshes.new_from_object(
                    obj.evaluated_get(context.evaluated_depsgraph_get()))
            finally:
                obj.modifiers.remove(modifier)
            size = triangles(candidate)
            if (0 < size < count
                    and all(a <= b for a, b in zip(topology(candidate), baseline))
                    and close_to_surface(candidate, original_tree, tolerance)
                    and close_to_surface(obj.data, surface_tree(candidate), tolerance)):
                if best is not None:
                    bpy.data.meshes.remove(best)
                best = candidate
                upper = ratio
                if size <= target:
                    break
            else:
                bpy.data.meshes.remove(candidate)
                lower = ratio
            if upper - lower < .015:
                break
            ratio = (upper + lower) / 2
        if best is not None:
            original = obj.data
            obj.data = best
            best = None
            if not original.users:
                bpy.data.meshes.remove(original)
    finally:
        if best is not None:
            bpy.data.meshes.remove(best)


def triangulate(mesh):
    bm = bmesh.new()
    try:
        bm.from_mesh(mesh)
        bmesh.ops.triangulate(bm, faces=list(bm.faces))
        bm.to_mesh(mesh)
    finally:
        bm.free()





