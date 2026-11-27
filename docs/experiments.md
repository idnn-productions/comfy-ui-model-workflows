# Experiments to come back to

Ideas that are not built. Each says what the question is, what we already know, and the first step.
Results of experiments that were run are in the [README](../README.md#experiments).

## 1. A Gaussian splat as the environment, meshes for everything else

**The question:** can a map's backdrop be a splat made from images, with our generated meshes placed inside it,
instead of building the whole environment out of meshes?

**Why it matters to us:** we build from images. A splat keeps the look of its source images, because nothing is
remeshed, unwrapped or baked, and those steps are where the mesh pipeline loses fidelity. A mesh is still needed
for anything that moves, is edited in Blender, is relit, or collides precisely.

**Where the idea comes from:** [isaac-mason/scifi-fetch-quest](https://github.com/isaac-mason/scifi-fetch-quest)
(MIT), a walkable splat world with a small quest. Read from its code and README on 2026-11-23; not run here.
How it is put together:

| part | what they do |
|---|---|
| The picture | One splat for the whole environment: a 163 MB `.spz` source, built into a 250 MB streaming level-of-detail `.rad` with the `build-lod` tool from the [Spark](https://github.com/sparkjsdev/spark) source repository. Rendered with Spark on Three.js. Game logic never touches it. |
| Collision | A separate, invisible triangle mesh aligned to the splat in world space. It feeds physics, the character controller and line-of-sight rays. Their comments call it hand-authored. |
| Navigation | A navmesh baked from the collider, flood-filled from a seed point so only the connected walkable interior is kept. |
| Characters and props | Ordinary glTF meshes placed inside the splat. |
| Lighting the meshes | Offline, the splat is rendered from a grid of points (every 1.5 m, about 4,600 cells) and the surrounding colour at each is stored as an order-2 spherical-harmonics probe in a 3D texture. At runtime each mesh samples it by world position, so a character near a red light turns red. Intensity (3.5) and saturation (2.0) are boosted at bake time because raw probe light reads dim and grey. |
| Shadows | A splat cannot receive shadows. An invisible catcher mesh drawn after the splats shows only the characters' shadows. |
| Configuration | Everything specific to the scene (asset paths, spawn points, probe settings, splat brightness) is in one file, so another world is a one-file edit. |

The repository does not say how its splat was made.

**What is open for us:**

- **Which generator.** None of our current tools makes a splat from reference views. This is the first thing to
  find out: what turns one image, or a few, into a splat of a place, and whether it runs on a 24 GB card.
- **The collider.** Theirs was made by hand. For a generated world it has to come from somewhere: the generator
  itself, a rough mesh of the same scene, or a mesh fitted to the splat.
- **Lighting our meshes from the splat.** The probe grid is the most reusable idea. Without it a generated model
  will look pasted in. Our textures also carry lighting painted into the base colour, which may fight it.
- **Scale and alignment.** Our models are finished to a length in metres. A splat has no scale of its own.
- **Costs.** A 250 MB download for one scene, lighting that cannot change, and a probe bake that needs real
  Chrome with a GPU (headless Chromium does not render splats faithfully).

**First step:** pick one image-to-splat generator, make a splat of a small room from one of our concept images,
and look at it in Spark's viewer. No collider, no editor work. That tells us whether the look is worth the rest.

**If the look holds, next:** a rough collider and navmesh for it, one of our generated models standing in it lit
by a probe grid, then how a map in the NFGames Map Editor would refer to a splat backdrop.

## 2. Smaller open questions from the mesh pipeline

- **More than one subject.** Every measurement so far is from one ship. The field-of-view results (auto for
  single view, 10 to 15 for multi-view) and the MoGe comparison need a character, an organic shape and a
  small prop before they are rules.
- **MoGe models on harder pictures.** The four models agreed within about four degrees on a perspective concept
  image but spread from 33 to 46 degrees on a flat side view. A subject where they disagree would show which to trust.
- **Hard edges after the clean-up.** AutoRemesher rounds crisp edges at low quad counts. Its `--sharp-edge`
  and `--adaptivity` options are passed through by `tools/blender_retopo.py` but have not been tried.
- **A better unwrap.** The clean-up uses Blender's Smart UV Project, which is quick and makes many small islands.
- **Rebuilding the workflows.** `tools/build_workflows.py` run against the current ComfyUI templates also changes
  a remesh setting (`sign_mode` from solid to udf). The committed workflows keep the tested value; the new one
  needs a comparison before a rebuild is committed.
- **A pass that repairs the model.** `tools/blender_inspect.py` now measures a result (welded piece and open-edge
  counts, floating pieces, flipped and zero-area faces) and the editor runs it, but nothing yet decides what to
  repair from those numbers before a person sees it.
