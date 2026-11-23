"""Score a generated mesh's proportions against orthographic reference views of the same subject.

    python tools/mesh_proportions.py --views <folder with left.png and front.png> model.glb [more.glb ...]

A single-view mesh comes out posed like its picture, so its size is measured along its own principal axes (longest,
middle, shortest), not along X, Y and Z. The reference is what the views show: length / height from the side view's
silhouette and width / height from the front view's. Prints one JSON line per model with both ratios and the error
in percent (0 is a perfect match). It reads files only.

This measures overall proportion, which is what a wrong field of view distorts. It says nothing about surface detail.
"""
import argparse
import json
import os
import struct

import numpy as np
from PIL import Image


def silhouette_box(path):
    """Width and height in pixels of what is not background (the colour of the picture's corners)."""
    rgb = np.asarray(Image.open(path).convert("RGB")).astype(np.int16)
    corners = np.stack([rgb[0, 0], rgb[0, -1], rgb[-1, 0], rgb[-1, -1]]).mean(axis=0)
    mask = np.abs(rgb - corners).sum(axis=2) > 40
    ys, xs = np.where(mask)
    return int(xs.max() - xs.min() + 1), int(ys.max() - ys.min() + 1)


def glb_vertices(path):
    with open(path, "rb") as f:
        data = f.read()
    length = struct.unpack_from("<I", data, 12)[0]
    doc = json.loads(data[20:20 + length])
    binary = data[20 + length + 8:]
    out = []
    for mesh in doc.get("meshes", []):
        for prim in mesh["primitives"]:
            acc = doc["accessors"][prim["attributes"]["POSITION"]]
            view = doc["bufferViews"][acc["bufferView"]]
            start = view.get("byteOffset", 0) + acc.get("byteOffset", 0)
            stride = view.get("byteStride", 12)
            raw = np.frombuffer(binary, dtype=np.uint8, count=(acc["count"] - 1) * stride + 12, offset=start)
            if stride == 12:
                out.append(raw.view("<f4").reshape(-1, 3))
            else:
                idx = (np.arange(acc["count"])[:, None] * stride + np.arange(12)[None, :]).ravel()
                out.append(raw[idx].copy().view("<f4").reshape(-1, 3))
    return np.concatenate(out).astype(np.float64)


def principal_extents(points):
    """Extents along the principal axes, longest first, over every vertex."""
    centred = points - points.mean(axis=0)
    _, _, axes = np.linalg.svd(centred[:: max(1, len(centred) // 200000)], full_matrices=False)
    along = centred @ axes.T
    lo, hi = along.min(axis=0), along.max(axis=0)
    return sorted((hi - lo).tolist(), reverse=True)


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("models", nargs="+")
    p.add_argument("--views", required=True)
    p.add_argument("--axes", choices=["principal", "xyz"], default="principal",
                   help="xyz for a mesh that stands level and faces +Z (a multi-view mesh): exact, where principal axes tilt a little")
    args = p.parse_args()
    lw, lh = silhouette_box(os.path.join(args.views, "left.png"))
    fw, fh = silhouette_box(os.path.join(args.views, "front.png"))
    want = {"length_over_height": lw / lh, "width_over_height": fw / fh}
    print(json.dumps({"reference": {k: round(v, 3) for k, v in want.items()}}))
    for path in args.models:
        points = glb_vertices(path)
        if args.axes == "xyz":
            size = points.max(axis=0) - points.min(axis=0)
            a, b, c = size[2], size[0], size[1]  # length along Z, width along X, height along Y
        else:
            a, b, c = principal_extents(points)
        got = {"length_over_height": a / c, "width_over_height": b / c}
        errors = {k: 100 * (got[k] / want[k] - 1) for k in want}
        print(json.dumps({"model": os.path.basename(path), **{k: round(v, 3) for k, v in got.items()},
                          "length_error_pct": round(errors["length_over_height"], 1), "width_error_pct": round(errors["width_over_height"], 1),
                          "error_pct": round(float(np.sqrt(np.mean(np.square(list(errors.values()))))), 1)}))


if __name__ == "__main__":
    main()
