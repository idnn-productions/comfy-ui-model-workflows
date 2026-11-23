"""Run one variation of a workflow on a running ComfyUI and report what came out: time, files, mesh proportions.

    python tools/experiment.py multiview --tag fov20 --set in_fov.value=20 --set in_use_ultrashape.value=false \
        --views asset3d/ship6 [--vertex-colours] [--out results.jsonl]
    python tools/experiment.py single --tag moge-auto --set MoGeInference.fov_x_degrees=0 --set in_image.image=asset3d/ship6/concept.png

--set is <node title>.<input>=<value> (JSON when it parses; a node without a title goes by its type). --views names a folder under ComfyUI's input folder with
front/left/back/right[/top/bottom].png. --vertex-colours also saves the full-resolution surface painted with vertex
colours, before decimation, unwrapping and baking (the mesh to clean up by hand, or with tools/blender_retopo.py).
Files are saved under ComfyUI's output folder as 3d/exp/<tag>*. One line of JSON is printed, and appended to --out.

It queues one prompt and waits. It refuses to start while ComfyUI is working on something else.
"""
import argparse
import json
import os
import struct
import sys
import time
import urllib.request

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(HERE)
sys.path.insert(0, HERE)
from queue_workflow import get, to_prompt  # noqa: E402

WORKFLOWS = {"multiview": "asset3d-pixal3d-multiview-ultrashape.json", "single": "asset3d-single-ultrashape.json"}


def glb_bounds(path):
    """Size of a GLB's geometry from its POSITION accessors' min/max, in the file's own frame (Y up, front +Z)."""
    with open(path, "rb") as f:
        magic, _, _ = struct.unpack("<III", f.read(12))
        if magic != 0x46546C67:
            return None
        length, kind = struct.unpack("<II", f.read(8))
        doc = json.loads(f.read(length))
    lo, hi = [float("inf")] * 3, [float("-inf")] * 3
    triangles = 0
    for mesh in doc.get("meshes", []):
        for prim in mesh.get("primitives", []):
            acc = doc["accessors"][prim["attributes"]["POSITION"]]
            if "min" in acc and "max" in acc:
                lo = [min(a, b) for a, b in zip(lo, acc["min"])]
                hi = [max(a, b) for a, b in zip(hi, acc["max"])]
            if "indices" in prim:
                triangles += doc["accessors"][prim["indices"]]["count"] // 3
    size = [h - l for l, h in zip(lo, hi)]
    return {"width_x": round(size[0], 4), "height_y": round(size[1], 4), "length_z": round(size[2], 4), "triangles": triangles,
            "vertex_colours": any("COLOR_0" in p["attributes"] for m in doc.get("meshes", []) for p in m["primitives"])}


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("workflow", choices=sorted(WORKFLOWS))
    p.add_argument("--tag", required=True)
    p.add_argument("--set", action="append", default=[], metavar="TITLE.INPUT=VALUE")
    p.add_argument("--views", help="folder under ComfyUI's input folder holding the views")
    p.add_argument("--vertex-colours", action="store_true")
    p.add_argument("--url", default="http://127.0.0.1:8188")
    p.add_argument("--comfy-output", required=True, help="ComfyUI's output folder")
    p.add_argument("--out", help="append the result line to this file")
    p.add_argument("--timeout", type=int, default=1800)
    args = p.parse_args()

    queue = get(args.url + "/queue")
    if queue["queue_running"] or queue["queue_pending"]:
        sys.exit("ComfyUI is working on something else: not starting")
    with open(os.path.join(REPO, "workflows", WORKFLOWS[args.workflow]), encoding="utf-8") as f:
        prompt, titles, _ = to_prompt(json.load(f), get(args.url + "/object_info"))

    def set_input(title, key, value):
        ids = titles.get(title) or [i for i, n in prompt.items() if n["class_type"] == title]  # a title, else a node type
        if not ids:
            sys.exit(f"no node titled {title}")
        for node_id in ids:
            prompt[node_id]["inputs"][key] = value

    if args.views:
        for view in ("front", "left", "back", "right", "top", "bottom"):
            if f"in_{view}_image" in titles:
                set_input(f"in_{view}_image", "image", f"{args.views}/{view}.png")
    applied = {}
    for item in args.set:
        target, _, raw = item.partition("=")
        title, _, key = target.rpartition(".")
        try:
            value = json.loads(raw)
        except ValueError:
            value = raw
        set_input(title, key, value)
        applied[target] = value
    prefix = f"3d/exp/{args.tag}"
    set_input("out_model", "filename_prefix", prefix)
    for label in ("textured", "clay"):
        set_input(f"out_preview_{label}", "filename_prefix", f"{prefix}_preview_{label}")
    if args.vertex_colours:
        # the surface the texture is baked from (after the remesh, or UltraShape when it is on), painted from the voxels
        surface = next(i for i, n in prompt.items() if n["_meta"]["title"] == "Switch: UltraShape")
        voxels = next(i for i, n in prompt.items() if n["class_type"] == "VaeDecodeTextureTrellis")
        prompt["exp_paint"] = {"class_type": "PaintMesh", "inputs": {"mesh": [surface, 0], "voxel_colors": [voxels, 0]}, "_meta": {"title": "exp: vertex colours"}}
        prompt["exp_file"] = {"class_type": "MeshToFile3D", "inputs": {"mesh": ["exp_paint", 0]}, "_meta": {"title": "exp: to file"}}
        save = dict(prompt[titles["out_model"][0]]["inputs"])
        save.update(model_3d=["exp_file", 0], filename_prefix=f"{prefix}_vcol")
        prompt["exp_save"] = {"class_type": "Save3DAdvanced", "inputs": save, "_meta": {"title": "exp: save vertex-colour mesh"}}

    request = urllib.request.Request(args.url + "/prompt", data=json.dumps({"prompt": prompt}).encode(), headers={"Content-Type": "application/json"})
    try:
        reply = json.load(urllib.request.urlopen(request, timeout=60))
    except urllib.error.HTTPError as e:
        sys.exit("ComfyUI rejected the prompt: " + e.read().decode()[:3000])
    prompt_id, started = reply["prompt_id"], time.time()
    while True:
        entry = get(f"{args.url}/history/{prompt_id}").get(prompt_id)
        if entry and entry.get("status", {}).get("status_str"):
            break
        if time.time() - started > args.timeout:
            sys.exit(f"still running after {args.timeout} s; prompt {prompt_id}")
        time.sleep(3)
    result = {"tag": args.tag, "workflow": args.workflow, "set": applied, "seconds": round(time.time() - started), "status": entry["status"]["status_str"]}
    if result["status"] != "success":
        result["error"] = [m[1].get("exception_message") for m in entry["status"].get("messages", []) if m[0] == "execution_error"]
    files = [item for out in entry.get("outputs", {}).values() for items in out.values() if isinstance(items, list)
             for item in items if isinstance(item, dict) and item.get("type") == "output" and item.get("filename")]
    result["files"] = [f"{i.get('subfolder', '')}/{i['filename']}" for i in files]
    for i in files:
        if i["filename"].endswith(".glb"):
            path = os.path.join(args.comfy_output, i.get("subfolder", ""), i["filename"])
            if os.path.isfile(path):
                key = "vertex_colour_mesh" if "_vcol" in i["filename"] else "model"
                result[key] = dict(glb_bounds(path) or {}, mb=round(os.path.getsize(path) / 1e6, 1), file=path)
    line = json.dumps(result)
    print(line)
    if args.out:
        with open(args.out, "a", encoding="utf-8") as f:
            f.write(line + "\n")


if __name__ == "__main__":
    main()
