# Setup and first run

Written and run on an RTX 4090 (24 GB), 64 GB RAM, Windows 11, ComfyUI 0.46.2 (Comfy Desktop), Python 3.13,
torch 2.12 + cu130. Timings and what we learned are at the end.

## 1. Our helper node

Copy `custom_nodes/asset3d_ultrashape/` into `ComfyUI/custom_nodes/` and restart ComfyUI. It adds
`Asset3D Shape Images` and `Asset3D UltraShape Refine`. Both workflows need it to load, even with
`in_use_ultrashape` off.

It has no dependencies of its own. The refine node needs UltraShape (next step) only when it actually runs.

## 2. UltraShape

1. **The UltraShape code.** The refine node looks for a folder `UltraShape-1.0` (holding `ultrashape/` and
   `configs/`) inside any pack under `ComfyUI/custom_nodes/`, or at the path in the environment variable
   `ASSET3D_ULTRASHAPE_DIR`. The copy maintained for Windows is the `UltraShape-1.0` folder of
   [ComfyUI-UltraShape1](https://github.com/jtydhr88/ComfyUI-UltraShape1). Either install that whole pack,
   or copy just that folder into `custom_nodes/asset3d_ultrashape/` (what we did: only our node loads, and
   the pack's own nodes stay out of the menu).
2. **Python packages.** UltraShape imports `trimesh`, `omegaconf`, `diffusers`, `accelerate`,
   `scikit-image` and `pytorch-lightning`. Do not install the pack's `requirements.txt` blindly: it is
   unpinned. Dry-run first (`uv pip install --dry-run ...`) and make sure torch and transformers are not
   replaced.
   - `cubvh` and `flash_attn` are optional. Without `flash_attn` a run can take 15 to 20 minutes instead of
     under a minute; there may be no wheel for torch 2.12.
   - UltraShape was written against transformers 4.37 and diffusers 0.30; we have 5.20 and 0.42. The names
     it imports exist in ours, but that is all that has been checked.
3. **The checkpoint.** `ultrashape_v1.pt` (7.4 GB, [infinith/UltraShape](https://huggingface.co/infinith/UltraShape))
   goes in `ComfyUI/models/UltraShape/`: the `models` folder of the ComfyUI install itself, not the shared
   models folder, because Comfy Desktop only maps its known subfolders to the shared one.
4. **DINOv2.** UltraShape loads `facebook/dinov2-large` (about 1.2 GB) from the Hugging Face cache and
   downloads it on first use if it is not there.

### What we installed

- `custom_nodes/asset3d_ultrashape/`, with `UltraShape-1.0/` inside it (from ComfyUI-UltraShape1 commit
  `1458c5d`).
- `pytorch-lightning` 2.6.6 (with `torchmetrics` and `lightning-utilities`; nothing else changed). The other
  packages were already there.
- The models listed below, `ultrashape_v1.pt` in `ComfyUI/models/UltraShape/`, DINOv2-large in the Hugging
  Face cache.

On Comfy Desktop the environment that runs is `ComfyUI/.venv`, not `standalone-env`: install packages there.
ComfyUI picks up the node at its next start.

Licence: Tencent Hunyuan 3D 2.1 Community License. It excludes the EU, the UK and South Korea.

## 3. Models

Sizes and URLs are in [models.json](models.json); folders are under `ComfyUI/models/`.

| file | folder | needed by |
|---|---|---|
| `pixal3d_multiview_int8_convrot.safetensors` | `diffusion_models` | multi-view |
| `pixal3d_int8_convrot.safetensors` | `diffusion_models` | single |
| `trellis_2_int8_convrot.safetensors` | `diffusion_models` | single |
| `trellis_2_shape_vae_bf16.safetensors`, `trellis_2_texture_vae_bf16.safetensors` | `vae` | both |
| `dino_v3_L_naf_fp32.safetensors` | `clip_vision` | both |
| `birefnet.safetensors` | `background_removal` | both |
| `moge_2_vitl_normal_fp16.safetensors` | `geometry_estimation` | single |
| `ultrashape_v1.pt` | `UltraShape` | both, when refining |

The single workflow loads both the Pixal3D and the TRELLIS.2 model file names, so both must exist even if
you only use one.

## 4. First-run checklist

Work up in steps, so that a failure points at one thing. Watch VRAM in Task Manager during each.

**Before starting:** in the NVIDIA Control Panel set *CUDA - Sysmem Fallback Policy* to *Prefer No Sysmem
Fallback* for ComfyUI's Python. An out-of-memory error then stops the run with a message instead of spilling
into system RAM and stalling the machine.

1. **Generation only.** Set `in_use_ultrashape` to false. Run the multi-view workflow at the defaults (shape
   resolution 1024). Expect a textured GLB. This is the stock template plus our loaders, so problems here are
   about inputs or models.
2. **Check the views.** Look at the saved `front_view`, `left_view`, ... crops: clean mattes, same scale. If
   left and right come out mirrored on the mesh, swap those two images.
3. **UltraShape at the safe settings.** `in_use_ultrashape` true, octree 384, 16384 latents, `low_vram` on.
   Things to look at, in this order:
   - *It fails to import or load.* Read the message; it names the missing piece (pack, dependency or
     checkpoint).
   - *The refined mesh is offset or the wrong size* against the vertex-colour preview: set `align` to
     `match_bounds`.
   - *Detail is refined against the wrong side* (for example the face detail appears on the back): step
     `yaw_degrees` through 90, 180, 270.
   - *The mesh renders inside out:* `flip_faces`.
   - *"index out of bounds":* lower `normalize_scale` to 0.95.
   - *Blank or smeared patches in the base colour:* the refined surface moved far from the coarse one there.
     Lower `guidance_scale`, or accept the stock mesh for that asset.
4. **Raise quality one step at a time:** UltraShape octree 512, then latents 32768, then
   `in_shape_resolution` 1536. Octree 768 and above are reported to need more than 24 GB.
5. **Try `in_shape_image_mode` = `greyscale`** on the same seeds and compare the geometry.

Record what you find here: measured VRAM, timings, and the settings that held up.

## Why the defaults are conservative

| setting | default | stock / authors' value | reason |
|---|---|---|---|
| `in_shape_resolution` | 1024 | 1536 | 1024 is upstream's low-VRAM tier; 1536 has out-of-memory reports on 24 GB |
| UltraShape octree | 384 | 1024 | about 8 GB at 384; 1024 ran out of memory on 24 GB for several users |
| UltraShape latents | 16384 | 32768 | a 12 GB card topped out near 17,000 |
| UltraShape `low_vram` | on | off | keeps one sub-model on the GPU at a time |

`Asset3D UltraShape Refine` also unloads ComfyUI's own models before it loads UltraShape and frees UltraShape
when it finishes, so the two never sit in VRAM together.

## Measured (RTX 4090, 2026-11-23)

Test subject: a generated six-crew starship, four orthographic views plus one three-quarter concept.

| run | time | peak VRAM (whole card) | result |
|---|---|---|---|
| single view, Pixal3D, shape 1024, no UltraShape | 77 s | 12.5 GB | textured GLB; correct colours; mesh tilted like the picture |
| four views, Pixal3D multi-view, shape 1024, no UltraShape | 72 s | 11.8 GB | level ship, right on every side; **colours shifted** (see below) |
| UltraShape alone, 12 steps, octree 384, 16384 latents | 64 s | about 10 GB | aligned with the input; cleaner, smoother surface |
| UltraShape alone, 50 steps, octree 512, 32768 latents | 357 s | 20.3 GB | 910k faces; sharper planes, small surface detail smoothed away |

| four views + UltraShape through the graph (50 steps, octree 384, 16384 latents) | 279 s | 12.7 GB | textured GLB on the refined mesh |

The two "UltraShape alone" rows were run in a separate Python process; the last row is the whole graph.

What we learned:

- **Build from the templates of the environment that runs.** ComfyUI runs `ComfyUI/.venv`; `standalone-env`
  holds an older template package whose MoGe node lacks `refine_steps`, and the server rejects the prompt.
- **UltraShape needs two fixes on our stack**, both now in place:
  - `pymeshlab` is "optional" upstream but named in function signatures, so the import fails without it. A
    stand-in `pymeshlab.py` sits beside the UltraShape code (`UltraShape-1.0/pymeshlab.py`).
  - The checkpoint names its DINOv2 layers the transformers 4 way; the node renames them for transformers 5.
- **Alignment works as designed**: `undo_normalize`, yaw 0, no face flip. The refined mesh lands on the coarse
  one.
- **flash_attn is not needed for usable speed** here: about 5 s per step at 16384 latents.
- **UltraShape is not a clear win on hard-surface subjects.** On the ship it straightened planes and edges
  but removed small panels and hatches that Pixal3D had modelled. Compare both per asset.
- **Baking colour onto the refined mesh needs the outer shell as reference.** With the stock reference (the
  raw decoded mesh, which has inner shells) the refined mesh baked with black patches. The graph now switches
  the bake reference to the remeshed outer shell when UltraShape is on, and the patches are gone.
- **UltraShape leaves small floating pieces**; the node drops them (`drop_floaters`).
- **Pixal3D multi-view shifts colours** (warm grey hull to olive, red to orange). The framed input views are
  correct and the stock texture wiring gives the same result, so it is the model, not the graph.
- **Projecting the reference views fixes every surface a view sees** (`in_use_view_projection`). Renders from
  rig cameras at yaw 0 / 90 / 180 / 270, distance `0.55 / tan(fov / 2)`, match the framed views to about 1 %
  in size and position, so the four pictures are painted straight onto the mesh. It adds about 15 s.
- **The top and bottom are still the generated texture.** Colour-matching it to the left view (measured in
  view space) brought the accent bands into line, but the hull on top stays greener than the projected
  sides: the generated tint varies across the ship, and one global correction cannot remove that. The real
  fix is a top (and bottom) reference view projected from pole cameras. Open.
- **Framing matters for long subjects.** `in_view_framing` = `match_height` fixed the scale of the nose-on
  views; the stock per-view crop would have enlarged them about 2.5 times.
