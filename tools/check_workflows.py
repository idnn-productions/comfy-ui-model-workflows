"""Static check of workflows/*.json. Reads files only; does not start ComfyUI or touch the GPU.

Checks that every link is consistent on both ends and joins matching types, that every plain (non-optional,
non-widget) input is connected, that widget values line up with their names, and, with --comfyui, that every
node type exists in that ComfyUI checkout or in this repo's custom_nodes.

    python tools/check_workflows.py --comfyui <path to ComfyUI>
"""
import argparse
import glob
import json
import os
import re
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(HERE)
UI_ONLY = {"MarkdownNote", "Note", "Reroute", "PrimitiveNode"}
NODE_ID = re.compile(r"""node_id\s*=\s*["']([A-Za-z0-9_]+)["']""")
MAPPING = re.compile(r"""^\s*["']([A-Za-z0-9_+ ]+)["']\s*:\s*[A-Za-z_][A-Za-z0-9_]*\s*,?\s*$""")


def known_node_types(comfyui):
    roots = [os.path.join(REPO, "custom_nodes")]
    if comfyui:
        roots += [os.path.join(comfyui, "nodes.py"), os.path.join(comfyui, "comfy_extras"),
                  os.path.join(comfyui, "comfy_api_nodes")]
    files = []
    for root in roots:
        if os.path.isfile(root):
            files.append(root)
        else:
            files += glob.glob(os.path.join(root, "**", "*.py"), recursive=True)
    known = set()
    for path in files:
        with open(path, encoding="utf-8", errors="replace") as f:
            text = f.read()
        known.update(NODE_ID.findall(text))
        if "NODE_CLASS_MAPPINGS" in text:
            block = text.split("NODE_CLASS_MAPPINGS", 1)[1]
            for line in block.splitlines()[:400]:
                m = MAPPING.match(line)
                if m:
                    known.add(m.group(1))
    return known


API_TITLES = {"in_front_image": "image", "in_left_image": "image", "in_back_image": "image", "in_right_image": "image",
              "in_top_image": "image", "in_bottom_image": "image", "in_use_top_bottom_views": "value",
              "in_use_ultrashape": "value", "in_target_faces": "target_face_count", "out_model": "filename_prefix",
              "out_preview_textured": "filename_prefix", "out_preview_clay": "filename_prefix"}


def check_pipeline():
    """pipeline.json (what the NFGames Map Editor reads): it parses, the files it names are here, and the API
    workflow has every input and output the editor sets, by node title."""
    errors = []
    try:
        with open(os.path.join(REPO, "pipeline.json"), encoding="utf-8") as f:
            pipeline = json.load(f)
    except (OSError, ValueError) as e:
        return [f"pipeline.json: {e}"]
    if pipeline.get("pipeline") != "asset3d" or pipeline.get("version") != 1:
        errors.append("pipeline.json: pipeline must be asset3d, version 1")
    for key in ("multiview", "finish", "cleanup"):
        if not os.path.isfile(os.path.join(REPO, str(pipeline.get(key, "")))):
            errors.append(f"pipeline.json: {key} names a file that is not here ({pipeline.get(key)})")
    try:
        with open(os.path.join(REPO, pipeline["multiview"]), encoding="utf-8") as f:
            graph = json.load(f)
    except (OSError, ValueError, KeyError):
        return errors or ["pipeline.json: the multiview workflow cannot be read"]
    by_title = {}
    for node in graph.values():
        by_title.setdefault(node.get("_meta", {}).get("title"), []).append(node)
    for title, name in API_TITLES.items():
        nodes = by_title.get(title, [])
        if len(nodes) != 1:
            errors.append(f"API workflow: {len(nodes)} nodes titled {title}, expected 1")
        elif name not in nodes[0].get("inputs", {}):
            errors.append(f"API workflow: {title} has no input {name}")
    return errors


def types_match(a, b):
    if "*" in (a, b):
        return True
    return bool(set(a.split(",")) & set(b.split(",")))


def check(path, known):
    with open(path, encoding="utf-8") as f:
        wf = json.load(f)
    errors = []
    nodes = {n["id"]: n for n in wf["nodes"]}
    if len(nodes) != len(wf["nodes"]):
        errors.append("duplicate node ids")
    links = {}
    for link in wf["links"]:
        if link[0] in links:
            errors.append(f"duplicate link id {link[0]}")
        links[link[0]] = link

    for link_id, (_, src, src_slot, dst, dst_slot, type_) in links.items():
        if src not in nodes or dst not in nodes:
            errors.append(f"link {link_id}: missing node {src} or {dst}")
            continue
        outs, ins = nodes[src].get("outputs", []), nodes[dst].get("inputs", [])
        if src_slot >= len(outs) or dst_slot >= len(ins):
            errors.append(f"link {link_id}: slot out of range")
            continue
        out, inp = outs[src_slot], ins[dst_slot]
        if link_id not in (out.get("links") or []):
            errors.append(f"link {link_id}: not listed on output {nodes[src]['type']}.{out['name']}")
        if inp.get("link") != link_id:
            errors.append(f"link {link_id}: input {nodes[dst]['type']}.{inp['name']} points elsewhere")
        if not types_match(out["type"], inp["type"]) or not types_match(out["type"], type_):
            errors.append(f"link {link_id}: {nodes[src]['type']}.{out['name']} ({out['type']}) -> "
                          f"{nodes[dst]['type']}.{inp['name']} ({inp['type']})")

    for node in wf["nodes"]:
        label = f"{node['type']} #{node['id']}"
        if known is not None and node["type"] not in UI_ONLY and node["type"] not in known:
            errors.append(f"{label}: unknown node type")
        for inp in node.get("inputs", []):
            link_id = inp.get("link")
            if link_id is not None and link_id not in links:
                errors.append(f"{label}: input {inp['name']} references missing link {link_id}")
            if link_id is None and inp.get("shape") != 7 and "widget" not in inp and node.get("mode", 0) == 0:
                errors.append(f"{label}: required input {inp['name']} is not connected")
        for out in node.get("outputs", []):
            for link_id in out.get("links") or []:
                if link_id not in links:
                    errors.append(f"{label}: output {out['name']} references missing link {link_id}")
        named = node.get("widgets_values_named")
        if named is not None and list(named.values()) != node.get("widgets_values"):
            errors.append(f"{label}: widgets_values and widgets_values_named disagree")

    if wf["last_node_id"] < max(nodes) or (links and wf["last_link_id"] < max(links)):
        errors.append("last_node_id / last_link_id are behind")
    titles = sorted(n["title"] for n in wf["nodes"] if n.get("title", "").startswith(("in_", "out_")))
    return errors, titles, len(nodes), len(links)


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--comfyui", help="ComfyUI checkout to resolve node types against")
    parser.add_argument("paths", nargs="*", default=sorted(glob.glob(os.path.join(REPO, "workflows", "*.json"))))
    args = parser.parse_args()
    args.explicit_only = any(a.endswith(".json") for a in sys.argv[1:])  # given files: check those alone
    known = known_node_types(args.comfyui) if args.comfyui else None
    failed = False
    for path in args.paths:
        errors, titles, n_nodes, n_links = check(path, known)
        print(f"{os.path.basename(path)}: {n_nodes} nodes, {n_links} links, "
              f"{'OK' if not errors else str(len(errors)) + ' problem(s)'}")
        print("  " + ", ".join(titles))
        for e in errors:
            print("  ERROR " + e)
        failed = failed or bool(errors)
    if not args.explicit_only:
        problems = check_pipeline()
        print(f"pipeline.json: {'OK' if not problems else str(len(problems)) + ' problem(s)'}")
        for e in problems:
            print("  ERROR " + e)
        failed = failed or bool(problems)
    sys.exit(1 if failed else 0)


if __name__ == "__main__":
    main()
