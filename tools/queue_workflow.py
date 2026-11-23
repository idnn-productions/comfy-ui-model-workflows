"""Queue one of the workflows on a running ComfyUI, with inputs set from the command line.

Converts the saved (canvas) workflow to the API prompt format using the server's own node definitions, applies
overrides addressed by node title, and posts it. Add --dry-run to stop before posting.

    python tools/queue_workflow.py workflows/asset3d-pixal3d-multiview-ultrashape.json \
        --set in_front_image.image=asset3d/ship6/front.png --set in_use_ultrashape.value=false

Overrides are <node title>.<input name>=<value>; values are parsed as JSON when possible (true, 12, 0.5),
otherwise kept as text. Image paths are relative to ComfyUI's input folder.
"""
import argparse
import json
import sys
import urllib.error
import urllib.request

UI_ONLY = {"MarkdownNote", "Note"}


def get(url):
    with urllib.request.urlopen(url, timeout=60) as r:
        return json.load(r)


def to_prompt(workflow, object_info):
    nodes = {n["id"]: n for n in workflow["nodes"]}
    links = {link[0]: link for link in workflow["links"]}
    prompt, titles = {}, {}
    for node in workflow["nodes"]:
        if node["type"] in UI_ONLY:
            continue
        if node.get("mode", 0) != 0:
            raise ValueError(f"{node['type']} #{node['id']} is muted or bypassed; this converter does not handle that")
        if node["type"] not in object_info:
            raise ValueError(f"node type {node['type']} is not registered on the server (restart ComfyUI?)")
        spec = object_info[node["type"]]["input"]
        known = set(spec.get("required", {})) | set(spec.get("optional", {}))
        inputs = {}
        for name, value in (node.get("widgets_values_named") or {}).items():
            # dotted names ("format.bit_depth") are sub-inputs of a dynamic combo and go through as they are
            if name in known or name.split(".")[0] in known:
                inputs[name] = value
        for inp in node.get("inputs", []):
            if inp.get("link") is not None:
                link = links[inp["link"]]
                inputs[inp["name"]] = [str(link[1]), link[2]]
        prompt[str(node["id"])] = {"class_type": node["type"], "inputs": inputs,
                                   "_meta": {"title": node.get("title") or node["type"]}}
        if node.get("title"):
            titles.setdefault(node["title"], []).append(str(node["id"]))
    for node_id, entry in prompt.items():
        for name, value in entry["inputs"].items():
            if isinstance(value, list) and len(value) == 2 and value[0] not in prompt:
                raise ValueError(f"node {node_id} input {name} comes from a node that is not in the prompt")
    return prompt, titles, nodes


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("workflow")
    parser.add_argument("--set", action="append", default=[], metavar="TITLE.INPUT=VALUE")
    parser.add_argument("--url", default="http://127.0.0.1:8188")
    parser.add_argument("--dry-run", action="store_true", help="convert and print the overrides, do not queue")
    parser.add_argument("--save-prompt", help="also write the API prompt to this file")
    args = parser.parse_args()

    with open(args.workflow, encoding="utf-8") as f:
        workflow = json.load(f)
    prompt, titles, _ = to_prompt(workflow, get(args.url + "/object_info"))

    for item in args.set:
        target, _, raw = item.partition("=")
        title, _, name = target.rpartition(".")
        if title not in titles:
            sys.exit(f"no node titled {title!r}. Titles: {', '.join(sorted(titles))}")
        try:
            value = json.loads(raw)
        except json.JSONDecodeError:
            value = raw
        for node_id in titles[title]:
            if name not in prompt[node_id]["inputs"]:
                sys.exit(f"{title} has no input {name!r}. It has: {', '.join(prompt[node_id]['inputs'])}")
            prompt[node_id]["inputs"][name] = value
            print(f"set {title}.{name} = {value!r}")

    if args.save_prompt:
        with open(args.save_prompt, "w", encoding="utf-8", newline="\n") as f:
            json.dump(prompt, f, indent=1)
    print(f"{len(prompt)} nodes in the prompt")
    if args.dry_run:
        return
    request = urllib.request.Request(args.url + "/prompt", data=json.dumps({"prompt": prompt}).encode(),
                                     headers={"Content-Type": "application/json"})
    try:
        with urllib.request.urlopen(request, timeout=60) as r:
            reply = json.load(r)
    except urllib.error.HTTPError as e:
        sys.exit("ComfyUI rejected the prompt:\n" + json.dumps(json.load(e), indent=1)[:6000])
    print("queued:", reply.get("prompt_id"), "node errors:", reply.get("node_errors") or "none")


if __name__ == "__main__":
    main()
