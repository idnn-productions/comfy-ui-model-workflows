# comfy-ui-model-workflows: notes for Claude

ComfyUI workflows and Blender scripts that turn reference views into a textured 3D asset. The README says how to use
them; `SETUP.md` says what to install. The NFGames Map Editor reads `pipeline.json` here and runs what it names.

## Working here

- The workflow files are generated: edit `tools/build_workflows.py`, not the JSON. `python tools/check_workflows.py`
  checks them.
- Each Blender script in `tools/` is one self-contained file, run headless:
  `blender --background --factory-startup --python tools/<script>.py -- <arguments>`. It prints one JSON line
  (`FINISH {…}`, `RETOPO {…}`, `INSPECT {…}`) that the editor reads. Keep every key that line has; only add.
- `pipeline.json` stays `"version": 1`. A new key must be optional for readers.
- Blender's glTF importer cuts a mesh along its UV seams. Count pieces and open edges on a welded copy.
- Anything that queues work on ComfyUI refuses to start while its queue is busy. Keep that.
- Commit messages are one plain sentence saying what is now true.

## Handoff, 2026-11-26

This is on `main` (merged and pushed 2026-11-27) and goes with the editor's `main`:

- `tools/blender_inspect.py` now counts loose pieces, open edges and non-manifold edges on the welded mesh, and adds
  `floating_pieces` (pieces with no vertex within 0.1% of the longest side of another piece's surface; `null` above
  500 pieces), `flipped_faces` and `zero_area_faces`. `--no-render` skips the pictures.
- `pipeline.json` names it as `inspect`. The editor runs it after it has saved a model and shows the numbers.
- `tools/test_inspect.py` tests it inside Blender:
  `blender --background --factory-startup --python-exit-code 1 --python tools/test_inspect.py` prints `INSPECT_TEST ok`.

On the PC: run that test, then follow the steps in the editor's `CLAUDE.md` (the editor's "Workflows folder" setting
must point at this checkout, pulled to `main`).
