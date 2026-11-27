"""Check blender_inspect.measure on small meshes built here and exported to GLB. Run with Blender, headless:

    blender --background --factory-startup --python-exit-code 1 --python tools/test_inspect.py

Prints `INSPECT_TEST ok` and exits 0, or names the failed assertion and exits 1.
"""
import os
import sys
import tempfile

import bmesh
import bpy

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import blender_inspect  # noqa: E402


def cube(name="cube", location=(0, 0, 0)):
    """A UV-unwrapped cube: the unwrap splits its vertices along the UV seams once exported."""
    bpy.ops.mesh.primitive_cube_add(size=2, location=location)
    obj = bpy.context.active_object
    obj.name = name
    bpy.ops.object.mode_set(mode="EDIT")
    bpy.ops.mesh.select_all(action="SELECT")
    bpy.ops.uv.smart_project()
    bpy.ops.object.mode_set(mode="OBJECT")
    return obj


def edit(obj, fn):
    bm = bmesh.new()
    bm.from_mesh(obj.data)
    bm.faces.ensure_lookup_table()
    fn(bm)
    bm.to_mesh(obj.data)
    bm.free()


def measure_scene(tmp):
    path = os.path.join(tmp, "t.glb")
    bpy.ops.export_scene.gltf(filepath=path, export_format="GLB")
    return blender_inspect.measure(blender_inspect.load(path))


def fresh():
    bpy.ops.wm.read_factory_settings(use_empty=True)


def check(label, stats, **expected):
    for key, want in expected.items():
        if stats[key] != want:
            print("INSPECT_TEST FAIL %s: %s is %r, expected %r" % (label, key, stats[key], want))
            sys.exit(1)


def main():
    with tempfile.TemporaryDirectory() as tmp:
        fresh()
        cube()
        check("unwrapped cube", measure_scene(tmp), loose_pieces=1, open_edges=0, non_manifold_edges=0,
              flipped_faces=0, floating_pieces=0, zero_area_faces=0)

        fresh()
        # a GLB holds triangles, so flip one triangle (not a quad, which would be two)
        edit(cube(), lambda bm: bmesh.ops.triangulate(bm, faces=bm.faces[:]))
        edit(bpy.context.active_object, lambda bm: bmesh.ops.reverse_faces(bm, faces=[bm.faces[0]]))
        check("cube with a flipped face", measure_scene(tmp), flipped_faces=1, open_edges=0)

        fresh()
        edit(cube(), lambda bm: bmesh.ops.delete(bm, geom=[bm.faces[0]], context="FACES_ONLY"))
        check("cube with a face deleted", measure_scene(tmp), open_edges=4, loose_pieces=1)

        fresh()
        cube("a")
        cube("b", location=(10, 0, 0))
        check("two cubes 10 apart", measure_scene(tmp), loose_pieces=2, floating_pieces=2, open_edges=0)
    print("INSPECT_TEST ok")


main()
