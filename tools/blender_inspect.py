"""Look at a GLB: print what it is made of and render it from fixed angles. Run with Blender, headless:

    blender --background --factory-startup --python tools/blender_inspect.py -- model.glb out_dir [--size 900]

Writes out_dir/<view>.png for six orthographic views and two perspective ones, and prints one JSON line of
measurements (faces, bounds, loose pieces, open edges, left/right symmetry error, textures).
"""
import json
import math
import os
import sys

import bmesh
import bpy
from mathutils import Vector
from mathutils.kdtree import KDTree


def load(path):
    bpy.ops.wm.read_factory_settings(use_empty=True)
    bpy.ops.import_scene.gltf(filepath=path)
    return [o for o in bpy.context.scene.objects if o.type == "MESH"]


def world_bounds(objects):
    points = [o.matrix_world @ Vector(c) for o in objects for c in o.bound_box]
    lo = Vector((min(p[i] for p in points) for i in range(3)))
    hi = Vector((max(p[i] for p in points) for i in range(3)))
    return lo, hi


def measure(objects):
    """Blender is Z up, and the glTF importer turns the file's Y-up frame into it: file +Y -> Blender +Z,
    file +Z (the asset's front) -> Blender -Y. Left/right is X in both."""
    lo, hi = world_bounds(objects)
    stats = {"objects": len(objects), "vertices": 0, "faces": 0, "triangles": 0, "loose_pieces": 0,
             "open_edges": 0, "non_manifold_edges": 0}
    points = []
    for obj in objects:
        bm = bmesh.new()
        bm.from_mesh(obj.data)
        stats["vertices"] += len(bm.verts)
        stats["faces"] += len(bm.faces)
        stats["triangles"] += sum(len(f.verts) - 2 for f in bm.faces)
        stats["open_edges"] += sum(1 for e in bm.edges if e.is_boundary)
        stats["non_manifold_edges"] += sum(1 for e in bm.edges if not e.is_manifold and not e.is_boundary)
        # connected pieces by flood fill
        seen = set()
        for v in bm.verts:
            if v.index in seen:
                continue
            stats["loose_pieces"] += 1
            stack = [v]
            seen.add(v.index)
            while stack:
                for e in stack.pop().link_edges:
                    for w in e.verts:
                        if w.index not in seen:
                            seen.add(w.index)
                            stack.append(w)
        step = max(1, len(bm.verts) // 60000)
        points += [obj.matrix_world @ v.co for v in list(bm.verts)[::step]]
        bm.free()
    size = hi - lo
    stats.update(bounds_min=[round(c, 4) for c in lo], bounds_max=[round(c, 4) for c in hi],
                 size_width_x=round(size.x, 4), size_length_y=round(size.y, 4), size_height_z=round(size.z, 4))
    # symmetry: mirror a sample of the surface in X about the bounding box's middle, distance to the surface
    middle = (lo.x + hi.x) / 2
    tree = KDTree(len(points))
    for i, p in enumerate(points):
        tree.insert(p, i)
    tree.balance()
    errors = sorted(tree.find(Vector((2 * middle - p.x, p.y, p.z)))[2] for p in points[::4])
    longest = max(size)
    stats["symmetry_error_median_pct"] = round(100 * errors[len(errors) // 2] / longest, 3)
    stats["symmetry_error_p95_pct"] = round(100 * errors[int(len(errors) * 0.95)] / longest, 3)
    stats["materials"] = sorted({m.name for o in objects for m in o.data.materials if m})
    stats["images"] = [{"name": i.name, "size": list(i.size)} for i in bpy.data.images if i.size[0]]
    return stats


def render_views(objects, out_dir, size):
    scene = bpy.context.scene
    scene.render.engine = "BLENDER_WORKBENCH"
    shading = scene.display.shading
    shading.light = "STUDIO"
    shading.color_type = "TEXTURE"
    shading.show_cavity = True
    shading.cavity_type = "BOTH"
    scene.render.resolution_x = scene.render.resolution_y = size
    scene.render.film_transparent = False
    scene.world = bpy.data.worlds.new("w")
    scene.world.color = (0.05, 0.05, 0.055)
    scene.view_settings.view_transform = "Standard"

    lo, hi = world_bounds(objects)
    centre, radius = (lo + hi) / 2, (hi - lo).length / 2
    cam_data = bpy.data.cameras.new("cam")
    cam = bpy.data.objects.new("cam", cam_data)
    scene.collection.objects.link(cam)
    scene.camera = cam
    # name: (direction the camera sits in, orthographic?)   front of the asset = Blender -Y
    views = {"front": ((0, -1, 0), True), "back": ((0, 1, 0), True), "left": ((1, 0, 0), True),
             "right": ((-1, 0, 0), True), "top": ((0, 0, 1), True), "bottom": ((0, 0, -1), True),
             "three_quarter_front": ((0.75, -1, 0.55), False), "three_quarter_back": ((-0.75, 1, 0.55), False)}
    for name, (direction, ortho) in views.items():
        d = Vector(direction).normalized()
        cam.location = centre + d * radius * 3.2
        up = "Y" if abs(d.z) < 0.99 else "Y"
        cam.rotation_euler = (-d).to_track_quat("-Z", up).to_euler()
        if abs(d.z) > 0.99:  # plan views: nose (Blender -Y) to the top of the picture
            cam.rotation_euler = (0, 0, math.pi) if d.z > 0 else (math.pi, 0, 0)
        cam_data.type = "ORTHO" if ortho else "PERSP"
        cam_data.ortho_scale = radius * 2.1
        cam_data.lens = 70
        cam_data.clip_end = radius * 20
        cam_data.clip_start = radius / 100
        scene.render.filepath = os.path.join(out_dir, name + ".png")
        bpy.ops.render.render(write_still=True)


def main():
    args = sys.argv[sys.argv.index("--") + 1:]
    path, out_dir = args[0], args[1]
    size = int(args[args.index("--size") + 1]) if "--size" in args else 900
    os.makedirs(out_dir, exist_ok=True)
    objects = load(path)
    stats = measure(objects)
    stats["file"] = os.path.basename(path)
    stats["file_mb"] = round(os.path.getsize(path) / 1e6, 1)
    print("INSPECT " + json.dumps(stats))
    render_views(objects, out_dir, size)


if __name__ == "__main__":
    main()
