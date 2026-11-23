"""Ask each MoGe model what horizontal field of view it sees in an image. No mesh is generated: this runs only the part
of the single-view workflow that ends at MoGeGeometryToFOV, which takes a few seconds.

    python tools/moge_fov.py asset3d/ship6/concept.png [more images under ComfyUI's input folder ...] [--models a b]

In the single-view workflow MoGe gives Pixal3D exactly one number, this field of view (Run MoGe Inference with
fov_x_degrees 0 recovers it from the picture; any other value is passed on as it is). So which MoGe model is used, and
what fov_x_degrees is set to, change the mesh only through this number.
"""
import argparse
import json
import os
import sys
import time
import urllib.request

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(HERE)
sys.path.insert(0, HERE)
from queue_workflow import get, to_prompt  # noqa: E402


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("images", nargs="+")
    p.add_argument("--models", nargs="*")
    p.add_argument("--url", default="http://127.0.0.1:8188")
    args = p.parse_args()
    info = get(args.url + "/object_info")
    spec = info["LoadMoGeModel"]["input"]["required"]["model_name"]
    models = args.models or (spec[0] if isinstance(spec[0], list) else spec[1]["options"])
    with open(os.path.join(REPO, "workflows", "asset3d-single-ultrashape.json"), encoding="utf-8") as f:
        full, titles, _ = to_prompt(json.load(f), info)
    fov = next(i for i, n in full.items() if n["class_type"] == "MoGeGeometryToFOV")
    keep, todo = {}, [fov]
    while todo:  # the nodes the field of view depends on, and nothing else
        i = todo.pop()
        if i in keep:
            continue
        keep[i] = full[i]
        todo += [v[0] for v in full[i]["inputs"].values() if isinstance(v, list) and len(v) == 2 and isinstance(v[0], str) and v[0] in full]
    loader = next(i for i, n in keep.items() if n["class_type"] == "LoadMoGeModel")
    image = titles["in_image"][0]
    rows = []
    for model in models:
        for name in args.images:
            prompt = json.loads(json.dumps(keep))
            prompt[loader]["inputs"]["model_name"] = model
            prompt[image]["inputs"]["image"] = name
            prompt["show"] = {"class_type": "PreviewAny", "inputs": {"source": [fov, 0]}, "_meta": {"title": "field of view"}}
            request = urllib.request.Request(args.url + "/prompt", data=json.dumps({"prompt": prompt}).encode(), headers={"Content-Type": "application/json"})
            try:
                prompt_id = json.load(urllib.request.urlopen(request, timeout=60))["prompt_id"]
            except urllib.error.HTTPError as e:
                sys.exit("ComfyUI rejected the prompt: " + e.read().decode()[:2000])
            started = time.time()
            while True:
                entry = get(f"{args.url}/history/{prompt_id}").get(prompt_id)
                if entry and entry.get("status", {}).get("status_str"):
                    break
                time.sleep(1)
            text = None
            for out in entry.get("outputs", {}).values():
                for value in out.values():
                    if isinstance(value, list) and value and isinstance(value[0], str):
                        text = value[0]
            rows.append({"model": model, "image": name, "fov_degrees": text, "seconds": round(time.time() - started, 1), "status": entry["status"]["status_str"]})
            print(json.dumps(rows[-1]), flush=True)


if __name__ == "__main__":
    main()
