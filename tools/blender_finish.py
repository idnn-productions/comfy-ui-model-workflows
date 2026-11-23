"""The fix pass every generated asset gets before a person sees it. Run with Blender, headless:

    blender --background --factory-startup --python tools/blender_finish.py -- in.glb out.glb \
        --length 40 [--name ship6] [--pivot centre|base] [--texture-format WEBP|JPEG|PNG] [--quality 90]

What it does, and why:
- joins the mesh into one named object with its transforms applied;
- drops disconnected pieces far smaller than the asset (generation leaves specks);
- scales the asset so its longest side is --length metres (generators work in a unit cube; glTF is metres),
  and puts the pivot at the centre of its footprint, at mid height or on the ground (--pivot);
- exports with compressed textures: a 4096 PNG set is tens of megabytes, WebP is a few.

It does not move vertices otherwise, re-unwrap or repaint: geometry and texture come from the pipeline.
Prints one JSON line (FINISH {...}) with before and after numbers.
"""
import json
import os
import sys

import bmesh
import bpy
from mathutils import Vector


def arg(args, flag, default=None):
    return args[args.index(flag) + 1] if flag in args else default


def bounds(obj):
    points = [obj.matrix_world @ Vector(c) for c in obj.bound_box]
    return (Vector((min(p[i] for p in points) for i in range(3))),
            Vector((max(p[i] for p in points) for i in range(3))))


def drop_specks(obj, fraction=0.002):
    """Delete connected pieces whose face count is below `fraction` of the largest piece's. Pieces are found on
    a welded copy of the topology (the mesh itself keeps its UV-seam splits)."""
    bm = bmesh.new()
    bm.from_mesh(obj.data)
    bm.verts.ensure_lookup_table()
    # union vertices that share a position, so UV seams do not count as gaps
    parent = list(range(len(bm.verts)))

    def find(i):
        while parent[i] != i:
            parent[i] = parent[parent[i]]
            i = parent[i]
        return i

    by_position = {}
    for v in bm.verts:
        key = (round(v.co.x, 6), round(v.co.y, 6), round(v.co.z, 6))
        if key in by_position:
            parent[find(v.index)] = find(by_position[key])
        else:
            by_position[key] = v.index
    for e in bm.edges:
        a, b = find(e.verts[0].index), find(e.verts[1].index)
        if a != b:
            parent[a] = b
    counts = {}
    for f in bm.faces:
        root = find(f.verts[0].index)
        counts[root] = counts.get(root, 0) + 1
    limit = max(counts.values()) * fraction
    doomed = [f for f in bm.faces if counts[find(f.verts[0].index)] < limit]
    removed_pieces = sum(1 for c in counts.values() if c < limit)
    if doomed:
        bmesh.ops.delete(bm, geom=doomed, context="FACES")
        bm.to_mesh(obj.data)
    bm.free()
    return len(counts), removed_pieces


def main():
    args = sys.argv[sys.argv.index("--") + 1:]
    src, dst = args[0], args[1]
    length = float(arg(args, "--length", 0) or 0)
    name = arg(args, "--name", os.path.splitext(os.path.basename(dst))[0])
    pivot = arg(args, "--pivot", "centre")
    texture_format = arg(args, "--texture-format", "WEBP")
    quality = int(arg(args, "--quality", 90))

    bpy.ops.wm.read_factory_settings(use_empty=True)
    bpy.ops.import_scene.gltf(filepath=src)
    meshes = [o for o in bpy.context.scene.objects if o.type == "MESH"]
    if not meshes:
        raise SystemExit("no mesh in " + src)
    bpy.ops.object.select_all(action="DESELECT")
    for o in meshes:
        o.select_set(True)
    bpy.context.view_layer.objects.active = meshes[0]
    if len(meshes) > 1:
        bpy.ops.object.join()
    obj = bpy.context.view_layer.objects.active
    # the importer may hang the mesh under empties: keep its world placement and drop them
    world = obj.matrix_world.copy()
    obj.parent = None
    obj.matrix_world = world
    for other in [o for o in bpy.context.scene.objects if o is not obj]:
        bpy.data.objects.remove(other, do_unlink=True)
    obj.name = obj.data.name = name
    bpy.ops.object.transform_apply(location=True, rotation=True, scale=True)

    before = {"triangles": sum(len(p.vertices) - 2 for p in obj.data.polygons), "vertices": len(obj.data.vertices),
              "size": [round(c, 4) for c in (bounds(obj)[1] - bounds(obj)[0])], "mb": round(os.path.getsize(src) / 1e6, 1)}

    pieces, removed = drop_specks(obj)

    lo, hi = bounds(obj)
    size = hi - lo
    factor = length / max(size) if length > 0 else 1.0
    centre = (lo + hi) / 2
    origin = Vector((centre.x, centre.y, lo.z if pivot == "base" else centre.z))
    for v in obj.data.vertices:
        v.co = (v.co - origin) * factor
    obj.data.update()
    bpy.context.view_layer.update()

    for material in obj.data.materials:
        if material:
            material.name = name

    bpy.ops.export_scene.gltf(filepath=dst, export_format="GLB", use_selection=False, export_apply=True,
                              export_image_format=texture_format, export_image_quality=quality,
                              export_yup=True, export_tangents=False, export_materials="EXPORT")
    lo, hi = bounds(obj)
    after = {"triangles": sum(len(p.vertices) - 2 for p in obj.data.polygons), "vertices": len(obj.data.vertices),
             "size": [round(c, 3) for c in (hi - lo)], "mb": round(os.path.getsize(dst) / 1e6, 1)}
    print("FINISH " + json.dumps({"name": name, "scale_factor": round(factor, 4), "pivot": pivot,
                                  "pieces": pieces, "specks_removed": removed, "before": before, "after": after}))


if __name__ == "__main__":
    main()
