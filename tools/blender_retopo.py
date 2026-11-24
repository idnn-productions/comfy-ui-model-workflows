"""Clean-up outside ComfyUI: a generated mesh in, a quad mesh with new UVs and baked maps out.
Run with Blender, headless:

    blender --background --factory-startup --python tools/blender_retopo.py -- source.glb out.glb \
        --quads 12000 [--remesher autoremesher|quadriflow] [--autoremesher <path to autoremesher(.exe)>] \
        [--texture 4096] [--image-format WEBP|JPEG|AUTO] [--length 40] [--name ship] [--sharp-edge 60] [--adaptivity 1.0]

What it does:
- remeshes the surface to about --quads quads, with AutoRemesher (https://github.com/huxingyi/autoremesher,
  MIT, run through its command line: the mesh goes out as OBJ and comes back as OBJ) or with Blender's own QuadriFlow;
- unwraps the new mesh (Smart UV Project). The maps are --texture pixels square; left out, as large as the
  source textures (4096 at most), or 2048 from vertex colours;
- bakes from the source onto it. From a textured model (what the workflows save as out_model): base colour,
  roughness, metallic and occlusion from its textures, and a tangent-space normal map of its surface with its own
  normal map in it. From a mesh with vertex colours and no texture (what tools/experiment.py --vertex-colours
  saves, or a mesh saved from ComfyUI right after Paint Mesh): base colour from the vertex colours, and the normal map;
- optionally scales it so its longest side is --length metres, and exports a GLB (--image-format AUTO keeps the
  textures as PNG, for a file that goes on to tools/blender_finish.py).

Bake from the textured model when you can: its colours are the ones painted from the reference views. Vertex
colours are the generated texture before that correction.

Prints one JSON line (RETOPO {...}) with the numbers and the seconds each step took.
"""
import json
import os
import subprocess
import sys
import tempfile
import time

import bpy
import numpy as np
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


def welded_copy(obj):
    """A copy with vertices at one place made one: the importer cuts a mesh along its UV seams, and a remesher
    would see every cut as an open edge."""
    only(obj)
    bpy.ops.object.duplicate()
    copy = bpy.context.view_layer.objects.active
    bpy.ops.object.mode_set(mode="EDIT")
    bpy.ops.mesh.select_all(action="SELECT")
    bpy.ops.mesh.remove_doubles(threshold=max(bounds_size(obj)) * 1e-6)
    bpy.ops.object.mode_set(mode="OBJECT")
    return copy


def main():
    args = sys.argv[sys.argv.index("--") + 1:]
    src, dst = args[0], args[1]
    quads = int(arg(args, "--quads", 12000))
    remesher = arg(args, "--remesher", "autoremesher")
    exe = arg(args, "--autoremesher", os.environ.get("AUTOREMESHER", "autoremesher"))
    size = int(arg(args, "--texture", 0))  # 0: as large as the source textures, 2048 without any
    image_format = arg(args, "--image-format", "WEBP")
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
    textured = [m for m in dense.data.materials if m and m.use_nodes
                and any(n.type == "TEX_IMAGE" and n.image for n in m.node_tree.nodes)
                and any(n.type == "BSDF_PRINCIPLED" for n in m.node_tree.nodes)]
    source_kind = "textures" if textured else "vertex colours" if colours else "none"
    before = counts(dense)
    if size <= 0:
        sizes = [n.image.size[0] for m in textured for n in m.node_tree.nodes if n.type == "TEX_IMAGE" and n.image]
        size = min(4096, max(sizes)) if sizes else 2048
    lap("import")

    # ---- remesh
    work = tempfile.mkdtemp(prefix="retopo-")
    report = None
    if remesher == "autoremesher":
        obj_in, obj_out, report_file = (os.path.join(work, n) for n in ("in.obj", "out.obj", "report.txt"))
        shell = welded_copy(dense)
        bpy.ops.wm.obj_export(filepath=obj_in, export_selected_objects=True, export_materials=False, export_uv=False,
                              export_normals=False, export_colors=False, export_triangulated_mesh=True)
        bpy.data.objects.remove(shell, do_unlink=True)
        command = [exe, "--input", obj_in, "--output", obj_out, "--report", report_file, "--target-quads", str(quads)]
        for flag in ("--edge-scaling", "--sharp-edge", "--smooth-normal", "--adaptivity", "--anisotropy"):
            if flag in args:
                command += [flag, arg(args, flag)]
        try:
            run = subprocess.run(command, capture_output=True, text=True)
        except OSError as e:
            raise SystemExit("AutoRemesher could not be started (%s): %s" % (exe, e))
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
        low = welded_copy(dense)
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

    # ---- bake from the source
    scene = bpy.context.scene
    scene.render.engine = "CYCLES"
    scene.cycles.device = "CPU"
    scene.cycles.samples = 1
    scene.render.bake.use_selected_to_active = True
    scene.render.bake.margin = 8
    reach = max(bounds_size(dense)) * 0.02
    scene.render.bake.cage_extrusion = reach
    scene.render.bake.max_ray_distance = reach * 2

    # what the source shows is baked through an emission shader, one channel at a time
    surfaces = []  # (material, its output node, the socket that feeds the output's surface)
    if textured:
        for m in textured:
            out = next(n for n in m.node_tree.nodes if n.type == "OUTPUT_MATERIAL" and n.is_active_output)
            surfaces.append((m, out, out.inputs["Surface"].links[0].from_socket))
    else:
        source = bpy.data.materials.new("dense")
        source.use_nodes = True
        source.node_tree.nodes.clear()
        dense.data.materials.clear()
        dense.data.materials.append(source)

    def show(channel):
        """Make the source emit one channel: a Principled BSDF input, "Occlusion", or the vertex colours.
        False when the source has nothing for it."""
        if not textured:
            if channel != "Base Color" or not colours:
                return False
            tree = source.node_tree
            out, emit, attribute = (tree.nodes.new(t) for t in ("ShaderNodeOutputMaterial", "ShaderNodeEmission", "ShaderNodeVertexColor"))
            attribute.layer_name = colours
            tree.links.new(attribute.outputs["Color"], emit.inputs["Color"])
            tree.links.new(emit.outputs[0], out.inputs[0])
            return True
        found = False
        for m, out, _ in surfaces:
            tree = m.node_tree
            if channel == "Occlusion":  # the importer keeps occlusion on a group node of its own
                socket = next((n.inputs["Occlusion"] for n in tree.nodes if n.type == "GROUP" and "Occlusion" in n.inputs), None)
            else:
                socket = next(n for n in tree.nodes if n.type == "BSDF_PRINCIPLED").inputs[channel]
            emit = tree.nodes.new("ShaderNodeEmission")
            if socket is not None and socket.is_linked:
                tree.links.new(socket.links[0].from_socket, emit.inputs["Color"])
                found = True
            elif socket is not None and channel != "Occlusion":
                v = socket.default_value
                emit.inputs["Color"].default_value = tuple(v) if hasattr(v, "__len__") else (v, v, v, 1.0)
                found = True
            tree.links.new(emit.outputs[0], out.inputs["Surface"])
        return found

    def show_surface():
        for m, out, surface in surfaces:
            m.node_tree.links.new(surface, out.inputs["Surface"])

    material = bpy.data.materials.new(name)
    material.use_nodes = True
    nodes, links = material.node_tree.nodes, material.node_tree.links
    shader = next(n for n in nodes if n.type == "BSDF_PRINCIPLED")
    low.data.materials.append(material)

    def target(label, data=True):
        image = bpy.data.images.new(name + "_" + label, size, size, alpha=False)
        if data:
            image.colorspace_settings.name = "Non-Color"
        node = nodes.new("ShaderNodeTexImage")
        node.image = image
        return node

    def bake(kind, node):
        bpy.ops.object.select_all(action="DESELECT")
        dense.select_set(True)
        low.select_set(True)
        bpy.context.view_layer.objects.active = low
        nodes.active = node
        bpy.ops.object.bake(type=kind)

    def grey(node):
        pixels = np.empty(size * size * 4, dtype=np.float32)
        node.image.pixels.foreach_get(pixels)
        return pixels.reshape(-1, 4)[:, 0].copy()

    baked, channels = [], {}
    colour_node = orm_node = None
    if show("Base Color"):
        colour_node = target("basecolor", data=False)
        bake("EMIT", colour_node)
        baked.append("base colour")
    if textured:
        for channel in ("Occlusion", "Roughness", "Metallic"):
            if show(channel):
                scratch = target(channel.lower())
                bake("EMIT", scratch)
                channels[channel] = grey(scratch)
                image = scratch.image
                nodes.remove(scratch)
                bpy.data.images.remove(image)
                baked.append(channel.lower())
        if channels:  # packed the way glTF wants them: occlusion in red, roughness in green, metallic in blue
            orm_node = target("orm")
            packed = np.ones((size * size, 4), dtype=np.float32)
            for index, channel in enumerate(("Occlusion", "Roughness", "Metallic")):
                if channel in channels:
                    packed[:, index] = channels[channel]
            orm_node.image.pixels.foreach_set(packed.ravel())
        show_surface()
    lap("bake colour")
    normal_node = target("normal")
    bake("NORMAL", normal_node)
    baked.append("normal")
    lap("bake normal")

    if colour_node:
        links.new(colour_node.outputs["Color"], shader.inputs["Base Color"])
    if orm_node:
        split = nodes.new("ShaderNodeSeparateColor")
        links.new(orm_node.outputs["Color"], split.inputs[0])
        if "Roughness" in channels:
            links.new(split.outputs[1], shader.inputs["Roughness"])
        if "Metallic" in channels:
            links.new(split.outputs[2], shader.inputs["Metallic"])
        group = next((n.node_tree for m, _, _ in surfaces for n in m.node_tree.nodes if n.type == "GROUP" and "Occlusion" in n.inputs), None)
        if "Occlusion" in channels and group:
            occlusion = nodes.new("ShaderNodeGroup")
            occlusion.node_tree = group
            links.new(split.outputs[0], occlusion.inputs["Occlusion"])
    else:
        shader.inputs["Roughness"].default_value = 0.6
    bump = nodes.new("ShaderNodeNormalMap")
    links.new(normal_node.outputs["Color"], bump.inputs["Color"])
    links.new(bump.outputs["Normal"], shader.inputs["Normal"])
    for node in (colour_node, orm_node, normal_node):
        if node:
            node.image.pack()

    # ---- scale, export
    bpy.data.objects.remove(dense, do_unlink=True)
    factor = length / max(bounds_size(low)) if length > 0 else 1.0
    if factor != 1.0:
        for v in low.data.vertices:
            v.co *= factor
        low.data.update()
    only(low)
    bpy.ops.export_scene.gltf(filepath=dst, export_format="GLB", use_selection=True, export_apply=True,
                              export_image_format=image_format, export_image_quality=90, export_yup=True, export_tangents=False)
    lap("export")
    print("RETOPO " + json.dumps({"name": name, "remesher": remesher, "target_quads": quads, "baked_from": source_kind,
                                  "baked": baked, "before": before, "after": counts(low), "texture": size,
                                  "size": [round(c, 3) for c in bounds_size(low)], "mb": round(os.path.getsize(dst) / 1e6, 1),
                                  "seconds": took, "report": report}))


if __name__ == "__main__":
    main()
