"""Build the asset3d workflows from ComfyUI's own Pixal3D / TRELLIS.2 templates.

The two stock templates (package comfyui_workflow_templates_json) hold the generation graph that Comfy-Org
tested. This script copies them and adds: separate reference-view loaders, the optional stripped shape
images, a texture conditioning that always uses the colour views, the UltraShape refine branch with its
switch, in_*/out_* titles and the how-to notes. Only JSON is read and written; nothing is run on the GPU.

    python tools/build_workflows.py --templates-dir <site-packages>/comfyui_workflow_templates_json/templates
"""
import argparse
import copy
import json
import os

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(HERE)
CORE = {"cnr_id": "comfy-core", "ver": "0.46.2"}

ULTRASHAPE_DEFAULTS = {
    "checkpoint": "ultrashape_v1.pt", "steps": 50, "guidance_scale": 5.0, "octree_resolution": 384,
    "num_latents": 16384, "num_chunks": 8000, "refine_seed": 42, "dtype": "bfloat16", "low_vram": True,
    "normalize_scale": 0.99, "yaw_degrees": "0", "align": "undo_normalize", "flip_faces": False,
    "drop_floaters": 0.01,
}


class Graph:
    def __init__(self, data):
        self.d = copy.deepcopy(data)
        self.nodes = {n["id"]: n for n in self.d["nodes"]}

    # -- ids ---------------------------------------------------------------------------------------------
    def _next_node_id(self):
        self.d["last_node_id"] += 1
        return self.d["last_node_id"]

    def _next_link_id(self):
        self.d["last_link_id"] += 1
        return self.d["last_link_id"]

    # -- nodes -------------------------------------------------------------------------------------------
    def add(self, type_, pos, size, inputs=(), outputs=(), widgets=None, title=None, core=True, color=None):
        """inputs: (name, type) or (name, type, flag) with flag 'optional' or 'widget'.
        widgets: ordered {widget_name: value}."""
        node = {"id": self._next_node_id(), "type": type_, "pos": list(pos), "size": list(size), "flags": {},
                "order": 0, "mode": 0, "inputs": [], "outputs": [],
                "properties": {"Node name for S&R": type_}}
        if core:
            node["properties"].update(CORE)
        for spec in inputs:
            entry = {"name": spec[0], "type": spec[1], "link": None}
            flag = spec[2] if len(spec) > 2 else None
            if flag == "optional":
                entry = {"name": spec[0], "shape": 7, "type": spec[1], "link": None}
            elif flag == "widget":
                entry = {"name": spec[0], "type": spec[1], "widget": {"name": spec[0]}, "link": None}
            node["inputs"].append(entry)
        for name, type_out in outputs:
            node["outputs"].append({"name": name, "type": type_out, "links": []})
        if title:
            node["title"] = title
        if widgets is not None:
            node["widgets_values"] = list(widgets.values())
            node["widgets_values_named"] = dict(widgets)
        if color:
            node["color"], node["bgcolor"] = color
        self.d["nodes"].append(node)
        self.nodes[node["id"]] = node
        return node["id"]

    def note(self, pos, size, text, title=None):
        node = {"id": self._next_node_id(), "type": "MarkdownNote", "pos": list(pos), "size": list(size),
                "flags": {}, "order": 0, "mode": 0, "inputs": [], "outputs": [], "properties": {},
                "widgets_values": [text], "widgets_values_named": {"text": text},
                "color": "#222", "bgcolor": "#000"}
        if title:
            node["title"] = title
        self.d["nodes"].append(node)
        self.nodes[node["id"]] = node
        return node["id"]

    def remove(self, node_id):
        node = self.nodes[node_id]
        for inp in node.get("inputs", []):
            if inp.get("link") is not None:
                self.unlink(node_id, inp["name"])
        for slot, out in enumerate(node.get("outputs", [])):
            for link_id in list(out.get("links") or []):
                link = self._link(link_id)
                self.unlink(link[3], self.nodes[link[3]]["inputs"][link[4]]["name"])
        self.d["nodes"].remove(node)
        del self.nodes[node_id]

    def title(self, node_id, title):
        self.nodes[node_id]["title"] = title

    def set_widget(self, node_id, name, value):
        node = self.nodes[node_id]
        named = node["widgets_values_named"]
        if name not in named:
            raise KeyError(f"node {node_id} ({node['type']}) has no widget {name}")
        named[name] = value
        node["widgets_values"][list(named).index(name)] = value

    def widget_to_input(self, node_id, name, type_):
        """Expose a widget as a linkable input, the way the frontend stores it."""
        node = self.nodes[node_id]
        if not any(i["name"] == name for i in node["inputs"]):
            node["inputs"].append({"name": name, "type": type_, "widget": {"name": name}, "link": None})

    # -- links -------------------------------------------------------------------------------------------
    def _link(self, link_id):
        return next(link for link in self.d["links"] if link[0] == link_id)

    def _input(self, node_id, name):
        for slot, inp in enumerate(self.nodes[node_id]["inputs"]):
            if inp["name"] == name:
                return slot, inp
        raise KeyError(f"node {node_id} ({self.nodes[node_id]['type']}) has no input {name}")

    def _output(self, node_id, name):
        outs = self.nodes[node_id]["outputs"]
        if isinstance(name, int):
            return name, outs[name]
        for slot, out in enumerate(outs):
            if out["name"] == name:
                return slot, out
        raise KeyError(f"node {node_id} ({self.nodes[node_id]['type']}) has no output {name}")

    def unlink(self, node_id, input_name):
        _, inp = self._input(node_id, input_name)
        link_id = inp.get("link")
        if link_id is None:
            return
        link = self._link(link_id)
        self.nodes[link[1]]["outputs"][link[2]]["links"].remove(link_id)
        self.d["links"].remove(link)
        inp["link"] = None

    def link(self, src, src_output, dst, dst_input):
        """Connect src.src_output -> dst.dst_input, replacing whatever fed dst_input."""
        self.unlink(dst, dst_input)
        out_slot, out = self._output(src, src_output)
        in_slot, inp = self._input(dst, dst_input)
        link_id = self._next_link_id()
        self.d["links"].append([link_id, src, out_slot, dst, in_slot, out["type"]])
        if out.get("links") is None:
            out["links"] = []
        out["links"].append(link_id)
        inp["link"] = link_id
        return link_id

    def source_of(self, node_id, input_name):
        _, inp = self._input(node_id, input_name)
        link = self._link(inp["link"])
        return link[1], link[2]

    def group(self, title, bounding, color="#3f789e"):
        gid = max((g.get("id", 0) for g in self.d["groups"]), default=0) + 1
        self.d["groups"].append({"id": gid, "title": title, "bounding": list(bounding), "color": color,
                                 "flags": {}})

    # -- output ------------------------------------------------------------------------------------------
    def finish(self):
        # topological "order", as the frontend would number it
        deps = {nid: set() for nid in self.nodes}
        for link in self.d["links"]:
            deps[link[3]].add(link[1])
        order, done = 0, set()
        pending = sorted(self.nodes)
        while pending:
            ready = [nid for nid in pending if deps[nid] <= done]
            if not ready:
                raise ValueError("cycle in graph")
            for nid in ready:
                self.nodes[nid]["order"] = order
                order += 1
                done.add(nid)
            pending = [nid for nid in pending if nid not in done]
        self.d["links"].sort(key=lambda link: link[0])
        return self.d


# ---------------------------------------------------------------------------------------------------------
# shared edits (node ids are the same in both stock templates)
# ---------------------------------------------------------------------------------------------------------

def add_ultrashape(g, origin, front_image, bg_model):
    """Coarse mesh (after the stock remesh, node 241) -> decimate -> UltraShape -> switch -> the stock
    decimate (186) and the two high-poly bake inputs (233, 224). front_image = (node, output) of the cropped
    colour front view."""
    x, y = origin
    coarse = g.add("DecimateMesh", (x + 30, y + 70), (360, 90),
                   inputs=[("mesh", "MESH")], outputs=[("mesh", "MESH")],
                   widgets={"target_face_count": 500000, "placement_mode": "midpoint"},
                   title="UltraShape input (decimate coarse mesh)")
    matte = g.add("RemoveBackground", (x + 30, y + 220), (360, 80),
                  inputs=[("bg_removal_model", "BACKGROUND_REMOVAL"), ("image", "IMAGE")],
                  outputs=[("mask", "MASK")], title="UltraShape image mask")
    refine = g.add("Asset3DUltraShapeRefine", (x + 430, y + 70), (420, 470),
                   inputs=[("mesh", "MESH"), ("image", "IMAGE"), ("mask", "MASK", "optional")],
                   outputs=[("mesh", "MESH")], widgets=dict(ULTRASHAPE_DEFAULTS),
                   title="in_ultrashape", core=False)
    use = g.add("PrimitiveBoolean", (x + 890, y + 70), (270, 70), outputs=[("BOOLEAN", "BOOLEAN")],
                widgets={"value": True}, title="in_use_ultrashape", color=("#322", "#533"))
    switch = g.add("ComfySwitchNode", (x + 890, y + 190), (270, 110),
                   inputs=[("on_false", "MESH"), ("on_true", "MESH"), ("switch", "BOOLEAN", "widget")],
                   outputs=[("output", "MESH")], widgets={"switch": True}, title="Switch: UltraShape")
    g.note((x + 890, y + 340), (400, 190),
           "**in_use_ultrashape** true = refine the geometry with UltraShape 1.0 before decimating and baking.\n\n"
           "false = stock result. The refine node is then skipped, but ComfyUI still needs the "
           "`asset3d_ultrashape` node installed to load this graph.\n\n"
           "First run: keep octree 384 and low_vram on. See SETUP.md before raising them.")

    # The colour bake snaps each texel to a reference surface before reading the voxel colours. The stock
    # reference is the raw decoded mesh, which has inner shells with no colour; where UltraShape moved the
    # surface the nearest point is sometimes one of those, and the texel bakes black. With UltraShape on, snap
    # to the remeshed outer shell (241) instead. Measured: the black patches disappear.
    raw_mesh = g.source_of(147, "reference_mesh")
    reference = g.add("ComfySwitchNode", (x + 1200, y + 190), (270, 110),
                      inputs=[("on_false", "MESH"), ("on_true", "MESH"), ("switch", "BOOLEAN", "widget")],
                      outputs=[("output", "MESH")], widgets={"switch": True},
                      title="Switch: bake reference surface")
    g.link(*raw_mesh, reference, "on_false")
    g.link(241, "mesh", reference, "on_true")
    g.link(use, 0, reference, "switch")
    g.link(reference, "output", 147, "reference_mesh")

    g.link(241, "mesh", coarse, "mesh")
    g.link(coarse, "mesh", refine, "mesh")
    g.link(*front_image, refine, "image")
    g.link(bg_model, 0, matte, "bg_removal_model")
    g.link(*front_image, matte, "image")
    g.link(matte, "mask", refine, "mask")
    g.link(241, "mesh", switch, "on_false")
    g.link(refine, "mesh", switch, "on_true")
    g.link(use, 0, switch, "switch")
    g.link(switch, "output", 186, "mesh")       # low-poly path: decimate -> unwrap -> bake
    g.link(switch, "output", 233, "high_poly")  # ambient occlusion from the refined surface
    g.link(switch, "output", 224, "high_poly")  # normal map from the refined surface
    g.group("UltraShape refine (geometry only)", (x, y, 1500, 570), "#a1309b")
    return refine


def add_view_projection(g, views, fov):
    """Project the four reference views onto the textured mesh from the rig's own cameras.

    The multi-view model's texture stage shifts colours (measured: warm grey to olive, red to orange). The
    framed views are the ground truth for every surface they see, and the rig cameras are known: orbit yaw
    0 / 90 / 180 / 270 at eye level, distance pad * 0.5 / tan(fov / 2) from the centre of the unit cube.
    Measured: renders from these cameras match the framed views to about 1 % in size and position.

    Two passes. Pass 1 projects onto the generated texture. The generated texture is then colour-matched to
    that result, so the top and bottom, which no view sees, stop clashing with the sides; pass 2 projects
    onto the corrected texture. views = {name: (node, output)} of the framed colour views; fov = node id."""
    apply_generated = 210
    low_poly = g.source_of(apply_generated, "mesh")
    generated_color = g.source_of(apply_generated, "base_color")
    x, y = g.nodes[224]["pos"][0] + 520, g.nodes[224]["pos"][1] + 330

    batch = None
    for i, name in enumerate(("left", "back", "right")):
        node = g.add("ImageBatch", (x + 30, y + 70 + 90 * i), (200, 60),
                     inputs=[("image1", "IMAGE"), ("image2", "IMAGE")], outputs=[("IMAGE", "IMAGE")],
                     title=f"Views batch (+{name})")
        g.link(*(views["front"] if batch is None else (batch, 0)), node, "image1")
        g.link(*views[name], node, "image2")
        batch = node
    to_mask = g.add("ImageToMask", (x + 30, y + 350), (200, 60), inputs=[("image", "IMAGE")],
                    outputs=[("MASK", "MASK")], widgets={"channel": "red"}, title="Views: subject mask")
    threshold = g.add("ThresholdMask", (x + 30, y + 450), (200, 60), inputs=[("mask", "MASK")],
                      outputs=[("MASK", "MASK")], widgets={"value": 0.03})
    shrink = g.add("GrowMask", (x + 30, y + 550), (200, 90), inputs=[("mask", "MASK")],
                   outputs=[("MASK", "MASK")], widgets={"expand": -2, "tapered_corners": True},
                   title="Views: trim the edge")
    # With top and bottom views: six pictures and six cameras (the orbit's two pole cameras come last).
    four = batch
    for i, name in enumerate(("top", "bottom")):
        node = g.add("ImageBatch", (x - 210, y + 70 + 90 * i), (200, 60),
                     inputs=[("image1", "IMAGE"), ("image2", "IMAGE")], outputs=[("IMAGE", "IMAGE")],
                     title=f"Views batch (+{name})")
        g.link(batch, 0, node, "image1")
        g.link(*views[name], node, "image2")
        batch = node
    use_poles = g.add("PrimitiveBoolean", (x - 210, y + 260), (200, 70), outputs=[("BOOLEAN", "BOOLEAN")],
                      widgets={"value": True}, title="in_use_top_bottom_views", color=("#322", "#533"))
    pick_views = g.add("ComfySwitchNode", (x - 210, y + 370), (200, 110),
                       inputs=[("on_false", "IMAGE"), ("on_true", "IMAGE"), ("switch", "BOOLEAN", "widget")],
                       outputs=[("output", "IMAGE")], widgets={"switch": True}, title="Switch: 4 or 6 views")
    g.link(four, 0, pick_views, "on_false")
    g.link(batch, 0, pick_views, "on_true")
    g.link(use_poles, 0, pick_views, "switch")
    batch = pick_views
    g.link(batch, 0, to_mask, "image")
    g.link(to_mask, 0, threshold, "mask")
    g.link(threshold, 0, shrink, "mask")

    distance = g.add("ComfyMathExpression", (x + 270, y + 70), (300, 110),
                     inputs=[("values.a", "FLOAT,INT,BOOLEAN")],
                     outputs=[("FLOAT", "FLOAT"), ("INT", "INT"), ("BOOL", "BOOLEAN")],
                     widgets={"expression": "0.55 / tan(a * 3.141592653589793 / 360)"},
                     title="Rig camera distance from in_fov")
    camera = g.add("CreateCameraInfo", (x + 270, y + 220), (300, 330),
                   inputs=[("mode.distance", "FLOAT", "widget"), ("fov", "FLOAT", "widget")],
                   outputs=[("camera_info", "LOAD3D_CAMERA")],
                   widgets={"mode": "orbit", "mode.yaw": 0.0, "mode.pitch": 0.0, "mode.distance": 3.12,
                            "target_x": 0.0, "target_y": 0.0, "target_z": 0.0, "roll": 0.0, "fov": 20.0,
                            "zoom": 1.0, "camera_type": "perspective"},
                   title="Rig camera (front)")
    orbit = g.add("CreateCameraOrbit", (x + 270, y + 590), (300, 150),
                  inputs=[("camera_info", "LOAD3D_CAMERA")], outputs=[("camera_info", "LOAD3D_CAMERA")],
                  widgets={"count": 4, "yaw": 360.0, "pitch": 0.0, "poles": "none"},
                  title="Rig cameras: front, left, back, right")
    g.link(fov, 0, distance, "values.a")
    g.link(distance, "FLOAT", camera, "mode.distance")
    g.link(fov, 0, camera, "fov")
    g.link(camera, 0, orbit, "camera_info")
    orbit_poles = g.add("CreateCameraOrbit", (x - 210, y + 520), (200, 150),
                        inputs=[("camera_info", "LOAD3D_CAMERA")], outputs=[("camera_info", "LOAD3D_CAMERA")],
                        widgets={"count": 4, "yaw": 360.0, "pitch": 0.0, "poles": "top and bottom"},
                        title="Rig cameras + top, bottom")
    pick_cameras = g.add("ComfySwitchNode", (x - 210, y + 710), (200, 110),
                         inputs=[("on_false", "LOAD3D_CAMERA"), ("on_true", "LOAD3D_CAMERA"),
                                 ("switch", "BOOLEAN", "widget")],
                         outputs=[("output", "LOAD3D_CAMERA")], widgets={"switch": True},
                         title="Switch: 4 or 6 cameras")
    g.link(camera, 0, orbit_poles, "camera_info")
    g.link(orbit, 0, pick_cameras, "on_false")
    g.link(orbit_poles, 0, pick_cameras, "on_true")
    g.link(use_poles, 0, pick_cameras, "switch")
    orbit = pick_cameras

    def bake(pos, mesh, title):
        node = g.add("BakeTextureFromViews", pos, (300, 200),
                     inputs=[("mesh", "MESH"), ("images", "IMAGE"), ("camera_info", "LOAD3D_CAMERA", "optional"),
                             ("mask", "MASK", "optional"), ("resolution", "INT", "widget")],
                     outputs=[("base_color", "IMAGE"), ("confidence", "MASK")],
                     widgets={"resolution": 0, "blend_exponent": 4.0, "fill_unseen": True}, title=title)
        g.link(*mesh, node, "mesh")
        g.link(batch, 0, node, "images")
        g.link(orbit, 0, node, "camera_info")
        g.link(shrink, 0, node, "mask")
        g.link(288, 0, node, "resolution")
        return node

    def apply(pos, color, title):
        node = g.add("ApplyTextureToMesh", pos, (260, 150),
                     inputs=[("mesh", "MESH"), ("base_color", "IMAGE")]
                     + [(n, "IMAGE", "optional") for n in ("metallic", "roughness", "occlusion", "normal_map")],
                     outputs=[("mesh", "MESH")], title=title)
        g.link(*low_poly, node, "mesh")
        g.link(*color, node, "base_color")
        for name in ("metallic", "roughness", "occlusion", "normal_map"):
            g.link(*g.source_of(apply_generated, name), node, name)
        return node

    # Colour-correct the generated texture before projecting, so the surfaces no view sees (top, bottom) match
    # the projected ones. The correction is measured in view space, where like is compared with like: the
    # generated texture rendered from the left rig camera against the real left view, both on black. That
    # transform is then applied to the whole texture atlas (ColorTransfer "target_frame": frame 0 sets the
    # transform, frame 1 is the atlas). Matching atlas to atlas was too weak: unseen areas diluted it.
    side_camera = g.add("CreateCameraInfo", (x + 610, y + 70), (300, 330),
                        inputs=[("mode.distance", "FLOAT", "widget"), ("fov", "FLOAT", "widget")],
                        outputs=[("camera_info", "LOAD3D_CAMERA")],
                        widgets={"mode": "orbit", "mode.yaw": 90.0, "mode.pitch": 0.0, "mode.distance": 3.12,
                                 "target_x": 0.0, "target_y": 0.0, "target_z": 0.0, "roll": 0.0, "fov": 20.0,
                                 "zoom": 1.0, "camera_type": "perspective"},
                        title="Rig camera (left)")
    g.link(distance, "FLOAT", side_camera, "mode.distance")
    g.link(fov, 0, side_camera, "fov")
    side_render = g.add("RenderMesh", (x + 610, y + 440), (300, 200),
                        inputs=[("mesh", "MESH"), ("model_3d_info", "LOAD3D_MODEL_INFO", "optional"),
                                ("camera_info", "LOAD3D_CAMERA", "optional"), ("texture", "IMAGE", "optional")],
                        outputs=[("image", "IMAGE"), ("mask", "MASK")],
                        widgets={"mode": "texture", "width": 1024, "height": 1024, "background": "#000000"},
                        title="Generated texture seen from the left")
    g.link(apply_generated, "mesh", side_render, "mesh")
    g.link(side_camera, 0, side_render, "camera_info")
    to_atlas_size = g.add("ImageScale", (x + 610, y + 680), (300, 150),
                          inputs=[("image", "IMAGE"), ("width", "INT", "widget"), ("height", "INT", "widget")],
                          outputs=[("IMAGE", "IMAGE")],
                          widgets={"upscale_method": "bilinear", "width": 4096, "height": 4096, "crop": "disabled"},
                          title="Scale to atlas size (so it can batch)")
    g.link(side_render, "image", to_atlas_size, "image")
    g.link(288, 0, to_atlas_size, "width")
    g.link(288, 0, to_atlas_size, "height")
    pair = g.add("ImageBatch", (x + 950, y + 70), (220, 60),
                 inputs=[("image1", "IMAGE"), ("image2", "IMAGE")], outputs=[("IMAGE", "IMAGE")],
                 title="[side render, generated atlas]")
    g.link(to_atlas_size, 0, pair, "image1")
    g.link(*generated_color, pair, "image2")
    match = g.add("ColorTransfer", (x + 950, y + 170), (300, 180),
                  inputs=[("image_target", "IMAGE"), ("image_ref", "IMAGE")], outputs=[("IMAGE", "IMAGE")],
                  widgets={"method": "reinhard_lab", "source_stats": "target_frame",
                           "source_stats.target_index": 0, "strength": 1.0},
                  title="Match generated colours to the left view")
    g.link(pair, 0, match, "image_target")
    g.link(*views["left"], match, "image_ref")
    atlas = g.add("ImageFromBatch", (x + 950, y + 390), (220, 100), inputs=[("image", "IMAGE")],
                  outputs=[("IMAGE", "IMAGE")], widgets={"batch_index": 1, "length": 1},
                  title="Corrected atlas")
    g.link(match, 0, atlas, "image")
    corrected = apply((x + 950, y + 530), (atlas, 0), "Apply corrected generated texture")
    second = bake((x + 1290, y + 70), (corrected, "mesh"), "Project the views (onto corrected texture)")
    final = apply((x + 1290, y + 320), (second, "base_color"), "Apply projected texture")
    x += 340

    use = g.add("PrimitiveBoolean", (x + 950, y + 520), (270, 70), outputs=[("BOOLEAN", "BOOLEAN")],
                widgets={"value": True}, title="in_use_view_projection", color=("#322", "#533"))
    switch = g.add("ComfySwitchNode", (x + 950, y + 630), (270, 110),
                   inputs=[("on_false", "MESH"), ("on_true", "MESH"), ("switch", "BOOLEAN", "widget")],
                   outputs=[("output", "MESH")], widgets={"switch": True}, title="Switch: view projection")
    g.link(apply_generated, "mesh", switch, "on_false")
    g.link(final, "mesh", switch, "on_true")
    g.link(use, 0, switch, "switch")
    g.link(switch, "output", 260, "mesh")
    g.note((x + 1260, y + 70), (380, 330),
           "**in_use_view_projection** true = paint the four reference views straight onto the mesh, so every "
           "surface a view sees gets its real colours and detail. The generated texture fills what no view "
           "sees (top, bottom, inside the engines), colour-matched to the views.\n\n"
           "Needs all four views. The cameras follow `in_fov`.\n\n"
           "false = the model's generated texture only.")
    g.group("Project the reference views onto the mesh", (x - 340, y, 2020, 870), "#b58b2a")


def add_preview_renders(g, prefix):
    """Four turntable pictures of the delivered mesh (textured) and of the high-poly surface (clay), saved
    next to the GLB, so a run can be judged without opening a 3D viewer."""
    final_mesh = g.source_of(next(n["id"] for n in g.d["nodes"] if n["type"] == "Save3DAdvanced"), "model_3d")
    final_mesh = g.source_of(final_mesh[0], "mesh")          # MeshToFile3D <- textured low-poly mesh
    high_poly = g.source_of(224, "high_poly")                # what the normal map is baked from
    x, y = g.nodes[224]["pos"][0] - 900, g.nodes[224]["pos"][1] + 330
    camera = g.add("CreateCameraInfo", (x + 30, y + 70), (300, 330),
                   inputs=[("mesh", "MESH", "optional")], outputs=[("camera_info", "LOAD3D_CAMERA")],
                   widgets={"mode": "orbit", "mode.yaw": 35.0, "mode.pitch": 20.0, "mode.distance": 1.9,
                            "target_x": 0.0, "target_y": 0.0, "target_z": 0.0, "roll": 0.0, "fov": 35.0,
                            "zoom": 1.0, "camera_type": "perspective"},
                   title="Preview camera")
    orbit = g.add("CreateCameraOrbit", (x + 360, y + 70), (270, 150),
                  inputs=[("camera_info", "LOAD3D_CAMERA")], outputs=[("camera_info", "LOAD3D_CAMERA")],
                  widgets={"count": 4, "yaw": 360.0, "pitch": 0.0, "poles": "none"}, title="Preview orbit")
    g.link(final_mesh[0], final_mesh[1], camera, "mesh")
    g.link(camera, "camera_info", orbit, "camera_info")
    for i, (label, mode, source) in enumerate((("textured", "auto", final_mesh), ("clay", "solid", high_poly))):
        render = g.add("RenderMesh", (x + 660, y + 70 + 250 * i), (300, 200),
                       inputs=[("mesh", "MESH"), ("model_3d_info", "LOAD3D_MODEL_INFO", "optional"),
                               ("camera_info", "LOAD3D_CAMERA", "optional"), ("texture", "IMAGE", "optional")],
                       outputs=[("image", "IMAGE"), ("mask", "MASK")],
                       widgets={"mode": mode, "width": 768, "height": 768, "background": "#202020"},
                       title=f"Preview render ({label})")
        save = g.add("SaveImage", (x + 990, y + 70 + 250 * i), (300, 220), inputs=[("images", "IMAGE")],
                     widgets={"filename_prefix": f"3d/{prefix}_preview_{label}"}, title=f"out_preview_{label}")
        g.link(source[0], source[1], render, "mesh")
        g.link(orbit, "camera_info", render, "camera_info")
        g.link(render, "image", save, "images")
    g.group("Preview renders", (x, y, 1320, 580), "#3f789e")


def common_titles(g, prefix):
    add_preview_renders(g, prefix)
    g.title(3, "in_seed_structure")
    g.title(18, "in_seed_shape")
    g.title(23, "in_seed_shape_hires")
    g.title(12, "in_seed_texture")
    g.title(94, "in_shape_resolution")
    g.title(186, "in_target_faces")
    g.title(288, "in_texture_resolution")
    # 1024 is upstream's low-VRAM tier; 1536 is the stock template value and the quality tier.
    g.set_widget(94, "target_resolution", 1024)
    save = next(n for n in g.d["nodes"] if n["type"] == "Save3DAdvanced")
    g.title(save["id"], "out_model")
    g.set_widget(save["id"], "filename_prefix", f"3d/{prefix}")


SETTINGS_NOTE = """## Settings worth knowing

| node | default | what it does |
|---|---|---|
| `in_shape_image_mode` | original | `greyscale` strips colour from the views that drive the **shape** stages. Texture always uses the colour views. Unproven either way: A/B it per asset. |
| `in_shape_resolution` | 1024 | Voxel resolution of the hi-res shape pass. 1536 is the quality tier and needs much more VRAM. |
| `in_use_ultrashape` | true | Refine geometry with UltraShape 1.0. |
| `in_ultrashape` | octree 384, 16384 latents, low_vram | Raise octree to 512, then latents to 32768, one at a time. |
| `in_target_faces` | 700000 | Faces of the delivered mesh. Detail above this lives in the baked normal map. |
| `in_texture_resolution` | 4096 | Texture atlas size. |
| `in_seed_*` | fixed | One seed per stage. Change `in_seed_shape` first if the shape is wrong. |

Output: `output/3d/` as GLB with base colour, metallic, roughness, ambient occlusion and a normal map baked
from the high-poly mesh.

**Not yet run.** These graphs were assembled from the stock templates without a test render. See SETUP.md
for the first-run checklist.
"""


# ---------------------------------------------------------------------------------------------------------
# workflow 1: four views -> Pixal3D multi-view
# ---------------------------------------------------------------------------------------------------------

MULTIVIEW_HOWTO = """# asset3d-pixal3d-multiview-ultrashape

Four reference views -> Pixal3D multi-view -> UltraShape -> baked PBR GLB.

## Inputs

Load one image per side into `in_front_image`, `in_left_image`, `in_back_image`, `in_right_image`.

The model assumes a fixed camera rig, so the views must be:

- exactly **90 degrees apart**, at **eye level** (no looking down or up),
- the **same scale and framing** in every view, one subject, nothing cropped off,
- taken with a **long lens** or rendered near-orthographic (`in_fov` 20 fits turnaround renders and most
  generated turnaround sheets; raise it for phone photos taken close up),
- evenly lit, without hard cast shadows. Shadows and stray background in the matte become geometry.

**left** means the subject's left side as the camera orbits (azimuth 90), **right** is azimuth 270. If the
sides come out mirrored, swap the two images.

Photos from arbitrary angles do not fit this model. Turn them into a clean four-view turnaround first (an
image-edit model does this well), or use `asset3d-single-ultrashape` with the best single view.

Fewer than four views: bypass (Ctrl+B) the unused loader together with its Remove Background, Crop and Save
nodes. The first connected view is treated as the front.

Backgrounds are removed with BiRefNet and the views are framed at 1024 with a 1.1 margin; the framed views
are saved as `front_view`, `left_view`, ... so you can check the mattes.

`in_view_framing` keeps the subject at one scale across the views. `match_height` (default) makes it equally
tall in every view, which also corrects views that were drawn at different sizes. `per_view` is the stock
behaviour (each view fills its frame) and is only right when height dominates every view, as on a standing
character.
"""


def build_multiview(template):
    g = Graph(template)

    # The stock graph slices one turnaround sheet; replace that with one loader per view.
    for node_id in (364, 374, 376):
        g.remove(node_id)
    views = [  # name, sheet crop node, RemoveBackground, ImageCropToMask, SaveImageAdvanced
        ("front", 338, 192, 312, 340), ("left", 341, 348, 349, 344),
        ("back", 342, 351, 350, 343), ("right", 345, 353, 352, 346),
    ]
    # The stock graph also crops every view to its own outline, which gives a long subject a different scale
    # in each view. One framing node replaces the four crops and keeps the scale shared.
    names = [v[0] for v in views]
    poles = ["top", "bottom"]
    frame = g.add("Asset3DFrameViews", (-2330, 370), (300, 460),
                  inputs=[(f"{n}_image", "IMAGE", "optional") for n in names]
                  + [(f"{n}_mask", "MASK", "optional") for n in names]
                  + [(f"{n}_image", "IMAGE", "optional") for n in poles]
                  + [(f"{n}_mask", "MASK", "optional") for n in poles],
                  outputs=[(n, "IMAGE") for n in names + poles],
                  widgets={"mode": "match_height", "size": 1024, "pad_factor": 1.1},
                  title="in_view_framing", core=False)
    saved = {}
    for name, crop, matte, crop_to_mask, save in views:
        y = g.nodes[matte]["pos"][1] - 40
        g.remove(crop)
        g.remove(crop_to_mask)
        loader = g.add("LoadImage", (-3360, y), (420, 400), outputs=[("IMAGE", "IMAGE"), ("MASK", "MASK")],
                       widgets={"image": f"{name}.png", "upload": "image"}, title=f"in_{name}_image")
        g.link(loader, "IMAGE", matte, "image")
        g.link(loader, "IMAGE", frame, f"{name}_image")
        g.link(matte, "mask", frame, f"{name}_mask")
        g.link(frame, name, save, "images")
        saved[name] = (save, "images")
    # Top and bottom views: texture projection only (the shape model takes the four sides).
    for i, name in enumerate(poles):
        y = 2170 + 440 * i
        loader = g.add("LoadImage", (-3360, y), (420, 400), outputs=[("IMAGE", "IMAGE"), ("MASK", "MASK")],
                       widgets={"image": f"{name}.png", "upload": "image"}, title=f"in_{name}_image")
        matte = g.add("RemoveBackground", (-2600, y + 40), (230, 80),
                      inputs=[("bg_removal_model", "BACKGROUND_REMOVAL"), ("image", "IMAGE")],
                      outputs=[("mask", "MASK")])
        save = g.add("SaveImage", (-2010, y + 40), (270, 270), inputs=[("images", "IMAGE")],
                     widgets={"filename_prefix": f"{name}_view"}, title=f"Framed {name} view")
        g.link(193, 0, matte, "bg_removal_model")
        g.link(loader, "IMAGE", matte, "image")
        g.link(loader, "IMAGE", frame, f"{name}_image")
        g.link(matte, "mask", frame, f"{name}_mask")
        g.link(frame, name, save, "images")
        saved[name] = (frame, name)
    g.group("Reference views (90 degrees apart, same scale)", (-3400, 250, 500, 1880), "#3f789e")
    g.group("Top and bottom views (nose up; texture only)", (-3400, 2140, 1700, 900), "#b58b2a")

    # Shape stages may see stripped views; the texture stage always sees the colour views.
    strip = g.add("Asset3DShapeImages", (-1630, 1600), (300, 150),
                  inputs=[(v, "IMAGE", "optional") for v in ("front", "left", "back", "right")],
                  outputs=[(v, "IMAGE") for v in ("front", "left", "back", "right")],
                  widgets={"mode": "original"}, title="in_shape_image_mode", core=False)
    texture_cond = g.add("Pixal3DMultiViewConditioning", (-1280, 1600), (310, 170),
                         inputs=[("clip_vision_model", "CLIP_VISION")]
                         + [(v, "IMAGE", "optional") for v in ("front", "left", "back", "right")]
                         + [("fov", "FLOAT", "widget")],
                         outputs=[("positive", "CONDITIONING"), ("negative", "CONDITIONING")],
                         widgets={"fov": 20}, title="Texture conditioning (colour views)")
    fov = g.add("PrimitiveFloat", (-1630, 1800), (300, 70), outputs=[("FLOAT", "FLOAT")],
                widgets={"value": 20.0}, title="in_fov")
    g.title(324, "Shape conditioning")
    g.widget_to_input(324, "fov", "FLOAT")
    for name in ("front", "left", "back", "right"):
        g.link(*saved[name], strip, name)
        g.link(strip, name, 324, name)
        g.link(*saved[name], texture_cond, name)
    g.link(15, 0, texture_cond, "clip_vision_model")
    g.link(fov, 0, 324, "fov")
    g.link(fov, 0, texture_cond, "fov")
    g.link(texture_cond, "positive", 98, "positive")
    g.link(texture_cond, "negative", 98, "negative")
    g.group("Shape / texture images", (-1650, 1550, 720, 350), "#8A8")

    add_ultrashape(g, origin=(1510, -420), front_image=saved["front"], bg_model=193)
    add_view_projection(g, saved, fov)
    common_titles(g, "asset3d-pixal3d-mv")

    g.note((-4320, 250), (860, 1000), MULTIVIEW_HOWTO, title="How to use")
    g.note((-4320, 1290), (860, 620), SETTINGS_NOTE, title="Settings")
    return g.finish()


# ---------------------------------------------------------------------------------------------------------
# workflow 2: one view -> Pixal3D or TRELLIS.2
# ---------------------------------------------------------------------------------------------------------

SINGLE_HOWTO = """# asset3d-single-ultrashape

One reference view -> Pixal3D **or** TRELLIS.2 -> UltraShape -> baked PBR GLB.

## Inputs

Load the best single view into `in_image`: the whole subject, a slight three-quarter angle or straight on,
long lens, even light, no hard cast shadow. If the file already has an alpha channel, set
`Switch: Remove background` to false to keep your own matte.

`in_use_trellis2`:

- **false = Pixal3D.** Pixel-aligned: the front of the mesh matches the picture closely. The camera field of
  view is estimated from the image (MoGe).
- **true = TRELLIS.2.** Not pixel-aligned; it invents more, which can look better on the unseen sides and
  worse on likeness. It is single-image only.

Run both on the same picture and keep the better one. With several good views of the subject, prefer
`asset3d-pixal3d-multiview-ultrashape`: the unseen sides are then measured instead of guessed.
"""


def build_single(template):
    g = Graph(template)
    g.remove(313)  # stock model-links note, replaced by README / models.json
    g.title(122, "in_image")
    g.set_widget(122, "image", "front.png")
    g.title(316, "in_use_trellis2")
    front = (302, "images")  # cropped colour view (PreviewImage passes it through)

    strip = g.add("Asset3DShapeImages", (-2620, 560), (300, 150),
                  inputs=[(v, "IMAGE", "optional") for v in ("front", "left", "back", "right")],
                  outputs=[(v, "IMAGE") for v in ("front", "left", "back", "right")],
                  widgets={"mode": "original"}, title="in_shape_image_mode", core=False)
    g.link(*front, strip, "front")
    g.link(strip, "front", 298, "image")
    g.link(strip, "front", 299, "image")
    g.title(298, "Shape conditioning (Pixal3D)")
    g.title(299, "Shape conditioning (TRELLIS.2)")

    # Texture conditioning from the colour view, with the same Pixal3D / TRELLIS.2 choice.
    x, y = -1760, 1290
    tex_pixal = g.add("Pixal3DConditioning", (x, y + 70), (260, 90),
                      inputs=[("clip_vision_model", "CLIP_VISION"), ("image", "IMAGE"),
                              ("camera_angle_x", "FLOAT", "widget")],
                      outputs=[("positive", "CONDITIONING"), ("negative", "CONDITIONING")],
                      widgets={"camera_angle_x": 49.13}, title="Texture conditioning (Pixal3D)")
    tex_trellis = g.add("Trellis2Conditioning", (x, y + 230), (260, 60),
                        inputs=[("clip_vision_model", "CLIP_VISION"), ("image", "IMAGE")],
                        outputs=[("positive", "CONDITIONING"), ("negative", "CONDITIONING")],
                        title="Texture conditioning (TRELLIS.2)")
    switches = []
    for i, polarity in enumerate(("positive", "negative")):
        sw = g.add("ComfySwitchNode", (x + 330, y + 70 + 160 * i), (270, 110),
                   inputs=[("on_false", "CONDITIONING"), ("on_true", "CONDITIONING"),
                           ("switch", "BOOLEAN", "widget")],
                   outputs=[("output", "CONDITIONING")], widgets={"switch": False},
                   title=f"Texture {polarity}")
        g.link(tex_pixal, polarity, sw, "on_false")
        g.link(tex_trellis, polarity, sw, "on_true")
        g.link(316, 0, sw, "switch")
        g.link(sw, "output", 98, polarity)
        switches.append(sw)
    clip_vision = g.source_of(298, "clip_vision_model")
    g.link(*clip_vision, tex_pixal, "clip_vision_model")
    g.link(*clip_vision, tex_trellis, "clip_vision_model")
    g.link(*front, tex_pixal, "image")
    g.link(*front, tex_trellis, "image")
    g.link(*g.source_of(298, "camera_angle_x"), tex_pixal, "camera_angle_x")
    g.group("Texture conditioning (colour view)", (x - 30, y, 660, 400), "#8A8")

    add_ultrashape(g, origin=(1560, -830), front_image=front, bg_model=193)
    common_titles(g, "asset3d-single")

    g.note((-5880, 440), (760, 620), SINGLE_HOWTO, title="How to use")
    g.note((-5880, 1100), (760, 620), SETTINGS_NOTE, title="Settings")
    return g.finish()


BUILDS = {
    "asset3d-pixal3d-multiview-ultrashape": ("3d_pixal3d_multi_views.json", build_multiview),
    "asset3d-single-ultrashape": ("3d_pixal3d_trellis2_image_to_model.json", build_single),
}


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--templates-dir", required=True,
                        help="folder holding ComfyUI's stock workflow templates (*.json)")
    parser.add_argument("--out-dir", default=os.path.join(REPO, "workflows"))
    args = parser.parse_args()
    os.makedirs(args.out_dir, exist_ok=True)
    for name, (template_file, build) in BUILDS.items():
        with open(os.path.join(args.templates_dir, template_file), encoding="utf-8") as f:
            workflow = build(json.load(f))
        path = os.path.join(args.out_dir, name + ".json")
        with open(path, "w", encoding="utf-8", newline="\n") as f:
            json.dump(workflow, f, indent=1, ensure_ascii=False)
            f.write("\n")
        print(f"{name}: {len(workflow['nodes'])} nodes, {len(workflow['links'])} links -> {path}")


if __name__ == "__main__":
    main()
