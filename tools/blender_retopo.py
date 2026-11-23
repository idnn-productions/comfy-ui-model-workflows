"""Clean-up outside ComfyUI: a dense mesh with vertex colours in, a quad mesh with UVs and baked maps out.
Run with Blender, headless:

    blender --background --factory-startup --python tools/blender_retopo.py -- dense.glb out.glb \
        --quads 15000 [--remesher autoremesher|quadriflow] [--autoremesher <path to autoremesher(.exe)>] \
        [--texture 2048] [--length 40] [--name ship] [--sharp-edge 60] [--adaptivity 1.0]

What it does:
- remeshes the dense surface to about --quads quads, with AutoRemesher (https://github.com/huxingyi/autoremesher,
  MIT, run through its command line: the mesh goes out as OBJ and comes back as OBJ) or with Blender's own QuadriFlow;
- unwraps the new mesh (Smart UV Project);
- bakes from the dense mesh onto it: base colour from the vertex colours, and a tangent-space normal map that carries
  the dense surface's detail;
- optionally scales it so its longest side is --length metres, and exports a GLB with WebP textures.

The dense mesh is what tools/experiment.py --vertex-colours saves (or a mesh saved from ComfyUI right after Paint
Mesh). Prints one JSON line (RETOPO {...}) with the numbers and the seconds each step took.
"""
import json
import os
import subprocess
import sys
import tempfile
import time

import bpy
from mathutils import Vector


def arg(args, flag, default=None):
    return args[args.index(flag) + 1] if flag in args else default


def only(obj):
    bpy.ops.object.select_all(action="DESELECT")
    obj.select_set(True)
    bpy.context.view_layer.objects.active = obj


def bounds_size(obj):
    points = [obj.matrix_world @ Vector(c) for c in obj.bound_box]
    return Vector((max(p[i] for p in points) - min(p[i] for p in points) for i in range(3)))


def counts(obj):
    polys = obj.data.polygons
    return {"faces": len(polys), "quads": sum(1 for p in polys if len(p.vertices) == 4),
            "triangles": sum(1 for p in polys if len(p.vertices) == 3), "vertices": len(obj.data.vertices)}


def main():
    args = sys.argv[sys.argv.index("--") + 1:]
    src, dst = args[0], args[1]
    quads = int(arg(args, "--quads", 15000))
    remesher = arg(args, "--remesher", "autoremesher")
    exe = arg(args, "--autoremesher", os.environ.get("AUTOREMESHER", "autoremesher"))
    size = int(arg(args, "--texture", 2048))
    length = float(arg(args, "--length", 0) or 0)
    name = arg(args, "--name", os.path.splitext(os.path.basename(dst))[0])
    took, clock = {}, time.time()

    def lap(label):
        nonlocal clock
        took[label] = round(time.time() - clock, 1)
        clock = time.time()

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
    dense = bpy.context.view_layer.objects.active
    world = dense.matrix_world.copy()
    dense.parent = None
    dense.matrix_world = world
    for other in [o for o in bpy.context.scene.objects if o is not dense]:
        bpy.data.objects.remove(other, do_unlink=True)
    only(dense)
    bpy.ops.object.transform_apply(location=True, rotation=True, scale=True)
    dense.name = "dense"
    colours = dense.data.color_attributes[0].name if dense.data.color_attributes else None
    before = counts(dense)
    lap("import")

    # ---- remesh
    work = tempfile.mkdtemp(prefix="retopo-")
    report = None
    if remesher == "autoremesher":
        obj_in, obj_out, report_file = (os.path.join(work, n) for n in ("in.obj", "out.obj", "report.txt"))
        only(dense)
        bpy.ops.wm.obj_export(filepath=obj_in, export_selected_objects=True, export_materials=False, export_uv=False,
                              export_normals=False, export_colors=False, export_triangulated_mesh=True)
        command = [exe, "--input", obj_in, "--output", obj_out, "--report", report_file, "--target-quads", str(quads)]
        for flag in ("--edge-scaling", "--sharp-edge", "--smooth-normal", "--adaptivity", "--anisotropy"):
            if flag in args:
                command += [flag, arg(args, flag)]
        run = subprocess.run(command, capture_output=True, text=True)
        if run.returncode != 0 or not os.path.isfile(obj_out):
            raise SystemExit("AutoRemesher failed (exit %s):\n%s" % (run.returncode, (run.stdout + run.stderr)[-2000:]))
        if os.path.isfile(report_file):
            with open(report_file, encoding="utf-8", errors="replace") as f:
                report = f.read()[:2000]
        existing = set(bpy.context.scene.objects)
        bpy.ops.wm.obj_import(filepath=obj_out)
        low = next(o for o in bpy.context.scene.objects if o not in existing and o.type == "MESH")
        only(low)
        bpy.ops.object.transform_apply(location=True, rotation=True, scale=True)
    else:
        only(dense)
        bpy.ops.object.duplicate()
        low = bpy.context.view_layer.objects.active
        for attr in list(low.data.color_attributes):
            low.data.color_attributes.remove(attr)
        low.data.materials.clear()
        bpy.ops.object.quadriflow_remesh(target_faces=quads, use_mesh_symmetry=False, use_preserve_sharp=True, use_preserve_boundary=True)
    low.name = low.data.name = name
    low.data.materials.clear()
    lap("remesh")

    # ---- unwrap
    only(low)
    bpy.ops.object.mode_set(mode="EDIT")
    bpy.ops.mesh.select_all(action="SELECT")
    bpy.ops.uv.smart_project(angle_limit=1.15, island_margin=0.003)
    bpy.ops.object.mode_set(mode="OBJECT")
    bpy.ops.object.shade_smooth()
    lap("unwrap")

    # ---- bake from the dense mesh
    scene = bpy.context.scene
    scene.render.engine = "CYCLES"
    scene.cycles.device = "CPU"
    scene.cycles.samples = 1
    scene.render.bake.use_selected_to_active = True
    scene.render.bake.margin = 8
    reach = max(bounds_size(dense)) * 0.02
    scene.render.bake.cage_extrusion = reach
    scene.render.bake.max_ray_distance = reach * 2

    source = bpy.data.materials.new("dense")
    source.use_nodes = True
    nodes, links = source.node_tree.nodes, source.node_tree.links
    nodes.clear()
    out = nodes.new("ShaderNodeOutputMaterial")
    emit = nodes.new("ShaderNodeEmission")
    links.new(emit.outputs[0], out.inputs[0])
    if colours:
        attribute = nodes.new("ShaderNodeVertexColor")
        attribute.layer_name = colours
        links.new(attribute.outputs["Color"], emit.inputs["Color"])
    dense.data.materials.clear()
    dense.data.materials.append(source)

    material = bpy.data.materials.new(name)
    material.use_nodes = True
    nodes, links = material.node_tree.nodes, material.node_tree.links
    shader = next(n for n in nodes if n.type == "BSDF_PRINCIPLED")
    colour_image = bpy.data.images.new(name + "_basecolor", size, size, alpha=False)
    normal_image = bpy.data.images.new(name + "_normal", size, size, alpha=False)
    normal_image.colorspace_settings.name = "Non-Color"
    colour_node = nodes.new("ShaderNodeTexImage")
    colour_node.image = colour_image
    normal_node = nodes.new("ShaderNodeTexImage")
    normal_node.image = normal_image
    low.data.materials.append(material)

    def bake(kind, node):
        bpy.ops.object.select_all(action="DESELECT")
        dense.select_set(True)
        low.select_set(True)
        bpy.context.view_layer.objects.active = low
        nodes.active = node
        bpy.ops.object.bake(type=kind)

    if colours:
        bake("EMIT", colour_node)
        lap("bake colour")
    bake("NORMAL", normal_node)
    lap("bake normal")
    if colours:
        links.new(colour_node.outputs["Color"], shader.inputs["Base Color"])
    bump = nodes.new("ShaderNodeNormalMap")
    links.new(normal_node.outputs["Color"], bump.inputs["Color"])
    links.new(bump.outputs["Normal"], shader.inputs["Normal"])
    shader.inputs["Roughness"].default_value = 0.6
    for image in (colour_image, normal_image):
        image.pack()

    # ---- scale, export
    bpy.data.objects.remove(dense, do_unlink=True)
    factor = length / max(bounds_size(low)) if length > 0 else 1.0
    if factor != 1.0:
        for v in low.data.vertices:
            v.co *= factor
        low.data.update()
    only(low)
    bpy.ops.export_scene.gltf(filepath=dst, export_format="GLB", use_selection=True, export_apply=True,
                              export_image_format="WEBP", export_image_quality=90, export_yup=True, export_tangents=False)
    lap("export")
    print("RETOPO " + json.dumps({"name": name, "remesher": remesher, "target_quads": quads, "vertex_colours": colours,
                                  "before": before, "after": counts(low), "texture": size,
                                  "size": [round(c, 3) for c in bounds_size(low)], "mb": round(os.path.getsize(dst) / 1e6, 1),
                                  "seconds": took, "report": report}))


if __name__ == "__main__":
    main()
