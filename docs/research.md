# Research: reference images to 3D assets

Researched 2026-11-23 from the model cards, papers, source code and issue trackers, and from the ComfyUI
0.46.2 source. This was written before the first run: nothing in it was measured by us. What we measured
afterwards is at the end of [SETUP.md](../SETUP.md).

Each claim is marked **[source]** (read in a primary source), **[report]** (a user's report in an issue or
README) or **[inference]** (our reasoning).

## Summary

1. ComfyUI runs Pixal3D, Pixal3D multi-view and TRELLIS.2 natively. No custom nodes or compiled extensions.
2. Pixal3D multi-view needs four views on a fixed rig. Photos from arbitrary angles do not fit it.
3. TRELLIS.2 is single-image only in every official path.
4. UltraShape refines geometry only. Texture has to be put back afterwards, and it needs a node pack.
5. Nobody has published a test of grey or "stripped" input images. It is a toggle, not a default.
6. Texture should come from the original colour images.
7. The out-of-memory risk on a 24 GB card sits in UltraShape's resolution setting and in the 1536 shape tier.

## The three models

### Pixal3D (TencentARC)

- Image-to-3D on the TRELLIS.2 backbone. It back-projects pixel features into 3D, so the front of the mesh
  matches the input image closely. Output is a mesh with base colour, metallic and roughness. **[source]**
  [model card](https://huggingface.co/TencentARC/Pixal3D), [repo](https://github.com/TencentARC/Pixal3D),
  [paper](https://arxiv.org/html/2605.10922)
- Licence: MIT for code and weights. **[source]**
- Native in ComfyUI since 0.34.0. **[source]**
  [Comfy blog](https://blog.comfy.org/p/trellis2-and-pixal3d-are-now-native),
  [tutorial](https://docs.comfy.org/tutorials/3d/pixal3d)
- **Multi-view.** The model needs known camera poses. ComfyUI's node fixes them: front, left, back and right,
  90 degrees apart, elevation 0, each view square, the subject at the same scale filling about 1/1.1 of the
  frame, field of view 20 degrees by default. The first connected view is the front. **[source]**
  `comfy_extras/nodes_trellis2.py`, `Pixal3DMultiViewConditioning`
- The maintainers say multi-view "requires the input views to have known and reasonably accurate camera
  parameters". One user got usable results with estimated poses within about 5 degrees. **[source]/[report]**
  [issue 9](https://github.com/TencentARC/Pixal3D/issues/9)
- Single-image mode assumes a small field of view; the stock workflow estimates it with MoGe. **[source]**
- Official defaults: 12 steps per stage; guidance 7.5 for structure and shape, 1.0 for texture; export at
  1,000,000 faces and a 4096 texture. The ComfyUI template reproduces these. **[source]**
- Known problems: double surfaces on thin shells, weak faces on people, out-of-memory reports including on
  24 GB with no maintainer answer. **[report]** [issues](https://github.com/TencentARC/Pixal3D/issues)
- On Toys4K the paper reports normal IoU 93.57 against 83.33 for Hunyuan3D-2.1. TRELLIS.2 was not a baseline,
  and the metric uses Pixal3D's own assumed projection. **[source]**

### TRELLIS.2-4B (Microsoft)

- Image-to-3D producing a mesh with base colour, metallic, roughness and opacity. Tiers 512, 1024 and 1536.
  MIT licence. **[source]** [model card](https://huggingface.co/microsoft/TRELLIS.2-4B),
  [repo](https://github.com/microsoft/TRELLIS.2)
- **Single image only.** The official pipeline takes one image. Multi-image support is an unmerged community
  pull request, and a user there reports results "fatter or thinner than expected". **[source]/[report]**
  [PR 104](https://github.com/microsoft/TRELLIS.2/pull/104)
- A community pack (visualbruno/ComfyUI-Trellis2) has multi-view and mesh-texturing nodes, but needs compiled
  wheels matched to the torch build; none are listed for our torch 2.12. **[source]**
  [repo](https://github.com/visualbruno/ComfyUI-Trellis2)
- Official preprocessing: crop to the alpha bounding box with no padding, composite on black, at most 1024 on
  the long side. Upscaling a reference beyond that buys nothing. **[source]**
- The raw mesh can have small holes; Comfy's own blog says TRELLIS.2 aligns to the input less well than
  Pixal3D. **[source]**

### UltraShape 1.0 (PKU-YuanGroup)

- A geometry refiner fine-tuned from Hunyuan3D-2.1: one image plus a coarse mesh in, a detailed mesh out.
  No UVs, no texture. **[source]** [repo](https://github.com/PKU-YuanGroup/UltraShape-1.0),
  [paper](https://arxiv.org/html/2512.21185v2)
- The coarse mesh does not have to come from Hunyuan3D. The loader only samples surface points and voxelises
  them at 128 cubed, after centring the bounding box and scaling it to fill the unit cube. It does not
  rotate the mesh, so the mesh must already match the image's orientation. **[source]** `surface_loaders.py`
- A user reports TRELLIS meshes work and that results get "wonkier" the more the mesh diverges from the
  image. **[report]** issue 14
- The paper: "inaccurate foreground segmentation or residual background and shadow artifacts can noticeably
  degrade geometry quality". **[source]**
- VRAM, all user-reported:

  | setting | VRAM |
  |---|---|
  | octree 384 | about 8 GB |
  | octree 512 | about 16 GB |
  | octree 768 | about 32 GB |
  | octree 1024, stock code | out of memory on 24 GB |
  | octree 1024, with the later memory fixes | 16 to 24 GB |

- Token count: 32768 is the authors' setting, 8192 the low-VRAM one. A 12 GB user found about 17,000 the
  ceiling. **[report]**
- Without flash-attn a 4090 user saw 15 to 20 minutes per run; with it, about 40 seconds. **[report]** issue 4
- Very dense input meshes were slow or broken; decimating to about 500,000 triangles fixed it. **[report]**
- Licence: **Tencent Hunyuan 3D 2.1 Community License.** It excludes the EU, the UK and South Korea, and
  needs a separate licence above 1M monthly active users. The Hugging Face card still says Apache-2.0; the
  conflict is an open, unanswered issue. **[source]** issues 11 and 25
- ComfyUI: [jtydhr88/ComfyUI-UltraShape1](https://github.com/jtydhr88/ComfyUI-UltraShape1) is the pack the
  authors point Windows users to. Its loader sits outside ComfyUI's model management. **[source]** `nodes.py`

## Input images

### Should the shape stage get a stripped or grey image?

- **For.** Hi3DGen attributes poor fine detail to "domain gaps and inherent ambiguities in RGB images" and
  inserts a normal-map step before geometry. **[source]** [paper](https://arxiv.org/abs/2503.22236) This is
  the only solid support, and it uses a trained normal estimator, not a grey render.
- **Against.** Pixal3D is pixel-aligned. An image-edit "clay pass" re-draws the picture and can move the
  silhouette, drop engraving or weave that is only visible as texture, and fill in thin or transparent
  parts. Whatever moves becomes geometry that no longer lines up with the originals. **[inference]**
- **Not found.** No published A/B of grey against colour input, and no documented clay pre-pass.
- Vendor guides say hard shadows get read as shape and highlights as material; one says shadows help.
  None of them test it.

Decision: a toggle (`in_shape_image_mode`). Plain greyscale removes colour without moving a single pixel, so
it keeps Pixal3D's alignment; it is the low-risk form of the idea.

### Should the texture stage get the original images?

Yes. Shape and texture are separate stages in all three models, and UltraShape's paper shows different images
per stage "without introducing conflicts". **[source]** Hunyuan3D 2.0 added a delighting module because
lighting gets baked into textures. **[source]** [paper](https://arxiv.org/html/2501.12202)

### Preparing the views

- **Matting.** BiRefNet is the default in ComfyUI's templates and won Cloudflare's independent comparison
  (IoU 0.87). **[source]** [post](https://blog.cloudflare.com/background-removal/) RMBG-2.0's lead is from its
  vendor's own benchmark and its weights are non-commercial.
- **Resolution.** About 1024 px across the subject is all the generators use.
- **Lens.** Long focal length or near-orthographic. UltraShape was trained on orthographic renders, and models
  that assume that distort meshes when given wide-angle pictures. **[source]**
  [Era3D](https://arxiv.org/abs/2405.11616)
- **Pose.** Limbs and accessories away from the body leave fewer holes. **[report]**
- **Real photos from several angles** differ in light, pose and lens. One good view plus a synthesised
  turnaround at the rig's angles is the safer input. **[inference]**

## What we could not find

- Official VRAM figures for Pixal3D or TRELLIS.2 at each tier.
- Any published workflow that chains UltraShape into a texturing stage.
- A neutral three-way benchmark of matting models.
- Evidence on which upscalers avoid inventing geometry cues.
