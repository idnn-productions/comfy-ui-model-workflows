"""One asset, start to finish: reference views in, a finished game-ready GLB with its references out.

    python tools/make_asset.py ship6 --views <folder with front/left/back/right[/top/bottom][/concept].png> \
        --length 40 --out <project>/models/ship6 --comfy-input <ComfyUI input folder>

Steps: copy the views into ComfyUI's input folder, queue the six-view workflow on the running ComfyUI and wait,
optionally clean the mesh up into quads (--quads, tools/blender_retopo.py with AutoRemesher), run the Blender fix
pass on the result (real-world scale, pivot, compressed textures), and write
<out>/<name>.glb with <out>/<name>.refs/ holding the reference images, which is the layout the NFGames Map
Editor's model viewer shows.

Nothing here judges the result. Look at it (tools/blender_inspect.py renders it) before showing it to anyone.
"""
import argparse
import json
import os
import shutil
import subprocess
import sys
import time
import urllib.request

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(HERE)
sys.path.insert(0, HERE)
from queue_workflow import get, to_prompt  # noqa: E402

SIDES = ("front", "left", "back", "right")
POLES = ("top", "bottom")
SOURCE_FACES = 200000  # triangles of the generated mesh when it is cleaned up into quads afterwards
BLENDER_DEFAULT = r"C:\Program Files\Blender Foundation\Blender 5.3\blender.exe"


def find_view(folder, name):
    for ext in (".png", ".jpg", ".jpeg", ".webp"):
        path = os.path.join(folder, name + ext)
        if os.path.isfile(path):
            return path
    return None


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("name", help="asset name: letters, digits, - and _")
    p.add_argument("--views", required=True, help="folder with the reference views")
    p.add_argument("--out", required=True, help="folder the finished <name>.glb and <name>.refs/ go into")
    p.add_argument("--comfy-input", required=True, help="ComfyUI's input folder")
    p.add_argument("--comfy-output", help="ComfyUI's output folder (default: next to the input folder)")
    p.add_argument("--length", type=float, required=True, help="the asset's longest side, in metres")
    p.add_argument("--faces", type=int, default=60000, help="triangles of the delivered mesh")
    p.add_argument("--pivot", choices=["centre", "base"], default="centre")
    p.add_argument("--quads", type=int, default=0, help="clean up into about this many quads with new UVs and re-baked maps (0: keep the generated triangles)")
    p.add_argument("--autoremesher", default=os.environ.get("AUTOREMESHER", "autoremesher"), help="the AutoRemesher program, for --quads")
    p.add_argument("--no-ultrashape", action="store_true")
    p.add_argument("--url", default="http://127.0.0.1:8188")
    p.add_argument("--blender", default=os.environ.get("BLENDER", BLENDER_DEFAULT))
    p.add_argument("--timeout", type=int, default=1800)
    args = p.parse_args()
    if not args.name.replace("-", "").replace("_", "").isalnum():
        sys.exit("name: letters, digits, - and _ only")

    views = {n: find_view(args.views, n) for n in SIDES + POLES + ("concept",)}
    missing = [n for n in SIDES if not views[n]]
    if missing:
        sys.exit(f"missing side views in {args.views}: {', '.join(missing)}")
    have_poles = all(views[n] for n in POLES)

    # 1. the views, where ComfyUI's loaders can see them
    sub = f"asset3d/{args.name}"
    dest = os.path.join(args.comfy_input, *sub.split("/"))
    os.makedirs(dest, exist_ok=True)
    for name, path in views.items():
        if path:
            shutil.copyfile(path, os.path.join(dest, name + os.path.splitext(path)[1].lower()))
    rel = lambda n: f"{sub}/{n}{os.path.splitext(views[n])[1].lower()}"

    # 2. generate
    with open(os.path.join(REPO, "workflows", "asset3d-pixal3d-multiview-ultrashape.json"), encoding="utf-8") as f:
        prompt, titles, _ = to_prompt(json.load(f), get(args.url + "/object_info"))

    def set_input(title, key, value):
        for node_id in titles[title]:
            prompt[node_id]["inputs"][key] = value

    for n in SIDES:
        set_input(f"in_{n}_image", "image", rel(n))
    for n in POLES:  # the loaders must point at a real file even when the views are not used
        set_input(f"in_{n}_image", "image", rel(n) if have_poles else rel("front"))
    set_input("in_use_top_bottom_views", "value", have_poles)
    set_input("in_use_ultrashape", "value", not args.no_ultrashape)
    # for a clean-up the generated mesh is only what the quads and their maps are made from: keep more of it
    set_input("in_target_faces", "target_face_count", max(args.faces, SOURCE_FACES) if args.quads else args.faces)
    prefix = f"3d/{args.name}"
    set_input("out_model", "filename_prefix", prefix)
    for label in ("textured", "clay"):
        set_input(f"out_preview_{label}", "filename_prefix", f"{prefix}_preview_{label}")

    request = urllib.request.Request(args.url + "/prompt", data=json.dumps({"prompt": prompt}).encode(),
                                     headers={"Content-Type": "application/json"})
    reply = json.load(urllib.request.urlopen(request, timeout=60))
    if reply.get("node_errors"):
        sys.exit("ComfyUI rejected the prompt: " + json.dumps(reply["node_errors"])[:3000])
    prompt_id = reply["prompt_id"]
    print(f"queued {prompt_id} ({'6' if have_poles else '4'} views, {args.faces} faces)")
    started = time.time()
    while True:
        entry = get(f"{args.url}/history/{prompt_id}").get(prompt_id)
        if entry and entry.get("status", {}).get("status_str"):
            break
        if time.time() - started > args.timeout:
            sys.exit(f"still running after {args.timeout} s; prompt {prompt_id}")
        time.sleep(4)
    if entry["status"]["status_str"] != "success":
        errors = [m[1] for m in entry["status"].get("messages", []) if m[0] == "execution_error"]
        sys.exit("generation failed: " + json.dumps(errors)[:3000])
    glbs = [item for out in entry["outputs"].values() for items in out.values() if isinstance(items, list)
            for item in items if isinstance(item, dict) and str(item.get("filename", "")).endswith(".glb")
            and item.get("type") == "output"]
    if not glbs:
        sys.exit("generation finished without a GLB in the output folder")
    comfy_output = args.comfy_output or os.path.join(os.path.dirname(os.path.normpath(args.comfy_input)), "output")
    raw = os.path.join(comfy_output, glbs[0].get("subfolder", ""), glbs[0]["filename"])
    print(f"generated in {time.time() - started:.0f} s: {raw}")

    # 3. the clean-up when asked for, the fix pass, 4. the finished asset with its references
    os.makedirs(args.out, exist_ok=True)
    final = os.path.join(args.out, args.name + ".glb")
    if args.quads:
        clean = os.path.join(args.out, args.name + ".clean.glb")
        run = subprocess.run([args.blender, "--background", "--factory-startup", "--python",
                              os.path.join(HERE, "blender_retopo.py"), "--", raw, clean, "--quads", str(args.quads),
                              "--autoremesher", args.autoremesher, "--image-format", "AUTO", "--name", args.name],
                             capture_output=True, text=True)
        report = next((line for line in run.stdout.splitlines() if line.startswith("RETOPO ")), None)
        if run.returncode != 0 or not report or not os.path.isfile(clean):
            sys.exit("Blender clean-up failed:\n" + (run.stdout + run.stderr)[-3000:])
        print(report)
        raw = clean
    run = subprocess.run([args.blender, "--background", "--factory-startup", "--python",
                          os.path.join(HERE, "blender_finish.py"), "--", raw, final, "--length", str(args.length),
                          "--name", args.name, "--pivot", args.pivot], capture_output=True, text=True)
    report = next((line for line in run.stdout.splitlines() if line.startswith("FINISH ")), None)
    if run.returncode != 0 or not report or not os.path.isfile(final):
        sys.exit("Blender fix pass failed:\n" + (run.stdout + run.stderr)[-3000:])
    print(report)
    if args.quads:
        os.remove(clean)
    refs = os.path.join(args.out, args.name + ".refs")
    os.makedirs(refs, exist_ok=True)
    for name, path in views.items():
        if path:
            shutil.copyfile(path, os.path.join(refs, name + os.path.splitext(path)[1].lower()))
    print(f"finished: {final}\nreferences: {refs}")


if __name__ == "__main__":
    main()
