# Plan: asset3d workflows

Written 2026-11-23 from [research.md](research.md). Status: **built, not yet run.**

## Goal

Turn reference images of one subject into a textured 3D asset (GLB with PBR maps) on our RTX 4090, with
Pixal3D or TRELLIS.2 generating and UltraShape 1.0 refining the geometry.

Constraint for this round: nothing was executed. The machine was being used remotely and could not be
allowed to hang, so the graphs were assembled and checked statically only.

## Approaches considered

| approach | verdict |
|---|---|
| **A. Extend ComfyUI's native Pixal3D / TRELLIS.2 templates** and add UltraShape as one node | **Chosen.** The generation graph is Comfy-Org's own, no compiled extensions, memory handled by ComfyUI. |
| B. Community packs for everything (visualbruno Trellis2, Saganaki Pixal3D, jtydhr88 UltraShape nodes) | Needs flash-attn, cumesh, o_voxel and similar wheels built for our torch 2.12; none published. Three packs loading models outside ComfyUI's memory management. |
| C. Call the official Python pipelines from scripts | Would allow arbitrary camera poses for Pixal3D, but it is Linux-first and gives up the ComfyUI graph. Worth revisiting only if four fixed views prove too limiting. |

## Pipeline

```
reference views -> BiRefNet matte -> crop 1024 (margin 1.1)
        |                                   |
        | colour views                      | optional stripped copies (in_shape_image_mode)
        v                                   v
  texture conditioning               shape conditioning
        |                                   |
        |                      structure -> shape 512 -> shape hi-res -> coarse mesh
        |                                   |                               |
        +---------> texture stage <---------+                    remesh, decimate to 500k
                         |                                                  |
                    voxel colours                       UltraShape refine (in_use_ultrashape)
                         |                                                  |
                         |                              high-poly mesh (refined or stock)
                         |                                   |                    |
                         |                         decimate, unwrap UVs     normal map + AO bake
                         v                                   v                    |
                   bake colour / metallic / roughness onto the low-poly mesh <----+
                                             |
                                            GLB
```

## Decisions

1. **Two workflows.** `asset3d-pixal3d-multiview-ultrashape` takes four views. `asset3d-single-ultrashape`
   takes one view and switches between Pixal3D and TRELLIS.2, because the stock single-image template already
   carries that switch and TRELLIS.2 has no multi-view mode.
2. **Texture after refinement, without a second texture model.** The generator's texture stage produces
   colours in a voxel volume around the *coarse* surface. `BakeTextureFromVoxel` takes a `reference_mesh`
   and snaps each texel onto it before sampling. Keeping the coarse mesh as that reference lets the colours
   land on the refined mesh even where UltraShape moved the surface.
3. **One wrapper node for UltraShape** (`Asset3DUltraShapeRefine`, in `custom_nodes/asset3d_ultrashape`). It
   takes and returns ComfyUI's native MESH, unloads ComfyUI's models before loading UltraShape, releases
   UltraShape afterwards, and puts the result back in the coarse mesh's position and scale by inverting
   UltraShape's normalisation. It imports the UltraShape code from the jtydhr88 pack rather than vendoring
   it, because of the licence.
4. **Stripped shape images as a toggle.** `Asset3DShapeImages` offers `original`, `greyscale` and
   `greyscale_autocontrast`. No edit-model clay pass is built in: it is unproven, would need another model,
   and risks breaking Pixal3D's pixel alignment. Clay images made elsewhere can still be loaded as the views.
5. **Conservative first-run defaults.** Shape tier 1024 (stock is 1536), UltraShape octree 384, 16384
   latents, low-VRAM on. Each is one widget to raise.
6. **Generated, not hand-edited.** `tools/build_workflows.py` rebuilds both workflows from the stock
   templates, so a ComfyUI update that changes the templates can be picked up by re-running it.

## Open questions for the first run

Listed in the order they will show up. Details and remedies are in [../SETUP.md](../SETUP.md).

1. Does `Asset3DUltraShapeRefine` load and run at all. It was written from source and never executed.
2. Is the refined mesh aligned with the coarse one (`align`) and facing the image (`yaw_degrees`).
3. Are there gaps in the baked colour where the refined surface moved far from the coarse one.
4. VRAM headroom at the defaults, then at shape 1536 and octree 512.
5. Whether `greyscale` shape images help or hurt, per asset type.

## Not built

- **Projection of the original photos onto the mesh** (`BakeTextureFromViews`). It needs a camera per image;
  worth adding for the four rig views once the base pipeline is proven.
- **Turnaround synthesis** from arbitrary photos. Do it with an image-edit model beforehand.
- **Delighting** before texturing.
- **A model downloader.** `models.json` uses the format of comfy-ui-video-workflows, whose
  `tools/get_models.py` can be copied over when the models are wanted.
