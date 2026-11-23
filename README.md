# comfy-ui-model-workflows

ComfyUI workflows that turn reference images of one subject into a textured 3D asset: Pixal3D or TRELLIS.2
generates the mesh and its PBR colours, UltraShape 1.0 refines the geometry, and the result is baked to a GLB
with base colour, metallic, roughness, ambient occlusion and a normal map.

> **Status: runs end to end on an RTX 4090** (2026-11-23, one test subject). Four views with UltraShape and
> view projection take about five minutes and peak near 13 GB of VRAM. Known weak spot: surfaces no reference
> view sees (top, bottom) keep the generated texture, whose colours drift. Measurements and open issues are
> in [SETUP.md](SETUP.md).

Background: [docs/research.md](docs/research.md) (what the models need and what is unproven) and
[docs/plan.md](docs/plan.md) (the design and why).

## Workflows

| workflow | input | generator |
|---|---|---|
| [`asset3d-pixal3d-multiview-ultrashape`](workflows/asset3d-pixal3d-multiview-ultrashape.json) | four views: front, left, back, right | Pixal3D multi-view |
| [`asset3d-single-ultrashape`](workflows/asset3d-single-ultrashape.json) | one view | Pixal3D or TRELLIS.2 (`in_use_trellis2`) |

Use the multi-view workflow when you have, or can make, a clean four-view turnaround: the unseen sides are
then measured instead of guessed. Use the single-view workflow for one good picture, and to compare Pixal3D
against TRELLIS.2 on it.

Each workflow carries **How to use** and **Settings** notes on the canvas.

## One command, start to finish

```
python tools/make_asset.py ship6 --views <folder of views> --length 40     --out <project>/models/ship6 --comfy-input <ComfyUI>/input
```

It takes a folder of reference views (`front`, `left`, `back`, `right`, optionally `top`, `bottom`, `concept`),
runs the six-view workflow on the running ComfyUI, then puts the result through the **fix pass** in Blender
(`tools/blender_finish.py`: real-world scale, pivot, small disconnected pieces removed, compressed textures) and
writes `<name>.glb` with a `<name>.refs/` folder of the reference images beside it. That is the layout the
NFGames Map Editor's model viewer shows, references included.

The intent: a generated model goes through the fix pass, and is looked at (`tools/blender_inspect.py` renders
eight fixed views and prints measurements), before a person is asked to review it.

Measured on the test ship: about four minutes, a 40 m, 60,000-triangle, 4 MB GLB (the raw output is 500,000
triangles and 36 MB at the default face target).

## Reference images

Pixal3D multi-view assumes a fixed camera rig. The four views must be 90 degrees apart at eye level, with the
subject at the same scale in each, shot with a long lens or rendered near-orthographic, evenly lit, with no
hard cast shadows. Photos taken from arbitrary angles do not fit; turn them into a four-view turnaround with
an image-edit model first, or use the single-view workflow.

Backgrounds are removed in the workflow (BiRefNet) and each view is cropped to 1024 px.

## Inputs and outputs

Inputs are nodes titled `in_*`, the result is `out_model` (GLB in `output/3d/`):

| node | meaning |
|---|---|
| `in_front_image`, `in_left_image`, `in_back_image`, `in_right_image` / `in_image` | reference views. `in_left_image` is the object seen from its own left side, so its front points to the left of the picture; `in_right_image` is the opposite |
| `in_top_image`, `in_bottom_image` (multi-view) | top and bottom views, drawn nose up; used to paint the texture, not for shape (`in_use_top_bottom_views`) |
| `in_fov` (multi-view) | field of view of the views; 20 fits turnaround renders |
| `in_view_framing` (multi-view) | how the views are scaled: `match_height` (default, same subject height in every view), `source_scale`, or `per_view` (stock) |
| `in_use_trellis2` (single) | false = Pixal3D, true = TRELLIS.2 |
| `in_shape_image_mode` | `original`, `greyscale` or `greyscale_autocontrast` for the shape stages only |
| `in_shape_resolution` | hi-res shape tier: 1024 (default) or 1536 |
| `in_use_ultrashape`, `in_ultrashape` | geometry refinement on/off and its settings |
| `in_use_view_projection` (multi-view) | paint the four reference views onto the mesh from the rig cameras; the generated texture fills only what no view sees |
| `in_target_faces` | faces of the delivered mesh (700,000) |
| `in_texture_resolution` | texture atlas size (4096) |
| `in_seed_structure`, `in_seed_shape`, `in_seed_shape_hires`, `in_seed_texture` | one seed per stage |

## Greyscale shape images

`in_shape_image_mode` strips colour from the views that drive the shape stages, on the idea that painted
detail and lighting are then less likely to be read as geometry. The texture stage always uses the colour
views. There is no published test of this either way, so it defaults to `original`: compare per asset.

## Setup

See [SETUP.md](SETUP.md). In short:

1. ComfyUI 0.46 or later (Pixal3D and TRELLIS.2 are core nodes).
2. Copy `custom_nodes/asset3d_ultrashape/` into `ComfyUI/custom_nodes/`.
3. For UltraShape: install the node pack
   [ComfyUI-UltraShape1](https://github.com/jtydhr88/ComfyUI-UltraShape1) and its requirements.
4. Download the models listed in [models.json](models.json).

## Rebuilding the workflows

The workflow files are generated from ComfyUI's own templates:

```
python tools/build_workflows.py --templates-dir <ComfyUI>/.venv/Lib/site-packages/comfyui_workflow_templates_json/templates
python tools/check_workflows.py --comfyui <ComfyUI>
```

Both only read and write JSON. Edit `build_workflows.py`, not the workflow files, or the next rebuild will
overwrite the change.

## Licences

Each model has its own licence; check before commercial use. Pixal3D and TRELLIS.2 are MIT. **UltraShape 1.0
is under the Tencent Hunyuan 3D 2.1 Community License**, which excludes the EU, the UK and South Korea. This
repository does not include UltraShape's code or weights. No licence has been chosen for this repository yet.

## Use from the NFGames Map Editor

The [NFGames Map Editor](https://github.com/idnn-productions/NFGames-Map-Editor)'s **New model** screen runs this
pipeline as its third step: it sends the approved views to your ComfyUI, waits, and finishes the result in Blender.

1. Set up ComfyUI as in [SETUP.md](SETUP.md), and install [Blender](https://www.blender.org/download/).
2. In the editor's Settings, under **3D model generation**, set the workflows folder to your checkout of this
   repository. (ComfyUI's address and Blender's location have defaults that usually fit.)
3. Step 3 of New model lists what it found: this folder, ComfyUI answering, the node types the workflow uses, the
   model files, and Blender.

What the editor reads here is named in [`pipeline.json`](pipeline.json): the workflow in ComfyUI's API format
(`workflows/api/`, exported from the canvas workflow with
`python tools/queue_workflow.py <workflow> --dry-run --save-prompt <file>`) and the Blender finish script. The
node pack also answers `GET /asset3d/status` on ComfyUI, which says whether UltraShape's code, Python packages
and checkpoint are in place.

## Licence

The code and workflows here are under the [MIT licence](LICENSE). The models they drive are not ours and have
their own licences, which apply to you when you download and use them; UltraShape's (Tencent Hunyuan 3D 2.1
Community License) excludes the EU, the UK and South Korea. See [SETUP.md](SETUP.md) and [models.json](models.json).
