"""asset3d_ultrashape: helper nodes for the asset3d workflows.

Asset3DFrameViews        cuts the reference views out and frames them all at one scale.
Asset3DShapeImages       optional "stripped" (greyscale) copies of the reference views for the shape stages.
Asset3DUltraShapeRefine  UltraShape 1.0 geometry refinement, native MESH in and native MESH out.

The refine node does not ship UltraShape. It imports the `ultrashape` library from an installed copy
(ComfyUI-UltraShape1 by jtydhr88, or any UltraShape-1.0 checkout named in ASSET3D_ULTRASHAPE_DIR) and
loads the model itself, so that it can free ComfyUI's models first and release its own afterwards.

UNTESTED: written from the UltraShape and ComfyUI sources on 2026-11-23, never executed. See SETUP.md.
"""
import gc
import logging
import math
import os
import sys

import numpy as np
import torch
from typing_extensions import override

import comfy.model_management
import comfy.utils
import folder_paths
from comfy_api.latest import ComfyExtension, IO, Types

LOG = logging.getLogger("asset3d_ultrashape")

MODEL_FOLDERS = ("UltraShape", "ultrashape")  # the two packs use different folder names
CHECKPOINT_EXTENSIONS = (".pt", ".ckpt", ".safetensors")
DEFAULT_CHECKPOINT = "ultrashape_v1.pt"
CONFIG_NAME = "infer_dit_refine.yaml"
LUMA = (0.2126, 0.7152, 0.0722)
SHAPE_MODES = ["original", "greyscale", "greyscale_autocontrast"]
VIEW_NAMES = ("front", "left", "back", "right")
POLE_NAMES = ("top", "bottom")  # used for texture projection only; the shape model takes the four sides


# ---------------------------------------------------------------------------------------------------------
# Shape-stage images
# ---------------------------------------------------------------------------------------------------------

def strip_image(image, mode):
    """image: [B, H, W, 3 or 4] in 0..1. Returns the same shape with colour removed (alpha is kept)."""
    if mode == "original":
        return image
    rgb = image[..., :3].float()
    weights = torch.tensor(LUMA, dtype=rgb.dtype, device=rgb.device)
    luma = (rgb * weights).sum(dim=-1, keepdim=True)
    if mode == "greyscale_autocontrast":
        # Stretch the subject's own brightness range so dark subjects keep contrast against the black
        # background. The subject is whatever is not (near) black; alpha is used when the image has it.
        subject = image[..., 3:4] > 0.5 if image.shape[-1] == 4 else luma > 0.02
        out = luma.clone()
        for b in range(luma.shape[0]):
            values = luma[b][subject[b]]
            if values.numel() < 16:
                continue
            lo = torch.quantile(values[:: max(1, values.numel() // 200000)], 0.01)
            hi = torch.quantile(values[:: max(1, values.numel() // 200000)], 0.99)
            if float(hi - lo) < 1e-4:
                continue
            stretched = ((luma[b] - lo) / (hi - lo)).clamp(0.0, 1.0) * 0.8 + 0.12
            out[b] = torch.where(subject[b], stretched, luma[b])
        luma = out
    grey = luma.expand(*luma.shape[:-1], 3).to(image.dtype)
    if image.shape[-1] == 4:
        return torch.cat([grey, image[..., 3:4]], dim=-1)
    return grey


class Asset3DShapeImages(IO.ComfyNode):
    @classmethod
    def define_schema(cls):
        return IO.Schema(
            node_id="Asset3DShapeImages",
            display_name="Asset3D Shape Images (strip colour)",
            category="3d/asset3d",
            description=(
                "Makes the copies of the reference views that drive the SHAPE stages. 'original' passes them "
                "through. The greyscale modes remove colour so that painted detail is less likely to be read as "
                "geometry; the texture stage keeps using the colour views. Connect only the views you have."
            ),
            inputs=[
                IO.Combo.Input("mode", options=SHAPE_MODES, default="original",
                               tooltip="original = no change. greyscale = luma only. greyscale_autocontrast = luma, "
                                       "stretched over the subject (helps dark subjects on the black background)."),
                *[IO.Image.Input(name, optional=True) for name in VIEW_NAMES],
            ],
            outputs=[IO.Image.Output(display_name=name) for name in VIEW_NAMES],
        )

    @classmethod
    def execute(cls, mode, front=None, left=None, back=None, right=None) -> IO.NodeOutput:
        views = (front, left, back, right)
        return IO.NodeOutput(*[None if v is None else strip_image(v, mode) for v in views])


# ---------------------------------------------------------------------------------------------------------
# Framing the four views to one scale
# ---------------------------------------------------------------------------------------------------------

FRAMING_MODES = ["match_height", "source_scale", "per_view"]


def mask_bbox(mask):
    """mask [H, W] -> (y0, y1, x0, x1), end-exclusive, of the pixels above 0.5."""
    ys, xs = torch.nonzero(mask > 0.5, as_tuple=True)
    if ys.numel() == 0:
        raise ValueError("Asset3DFrameViews: a view has an empty mask (no subject found)")
    return int(ys.min()), int(ys.max()) + 1, int(xs.min()), int(xs.max()) + 1


def frame_views(images, masks, mode, size, pad_factor):
    """images: list of [H, W, 3|4] in 0..1, masks: list of [H, W] foreground. Returns one [size, size, 3]
    per view: the subject on black, centred, all views at one scale.

    The multi-view rig expects the subject at the same scale in every view. Cropping each view to its own
    outline breaks that for anything long: a ship seen nose-on would be enlarged to fill the frame like its
    side view. A subject's height is the same from front, side and back, so match_height uses it as the ruler.
    """
    boxes = [mask_bbox(m) for m in masks]
    heights = [float(b[1] - b[0]) for b in boxes]
    widths = [float(b[3] - b[2]) for b in boxes]
    if mode == "match_height":
        factors = [max(heights) / h for h in heights]
    else:  # source_scale trusts the pictures; per_view is the stock behaviour
        factors = [1.0] * len(boxes)
    extents = [max(w, h) * f for w, h, f in zip(widths, heights, factors)]
    shared = max(extents)

    out, framed_sizes = [], []
    for image, mask, box, f, extent in zip(images, masks, boxes, factors, extents):
        k = size / (pad_factor * extent) if mode == "per_view" else f * size / (pad_factor * shared)
        out.append(place_view(image, mask, box, k, size))
        framed_sizes.append(((box[3] - box[2]) * k, (box[1] - box[0]) * k))  # (width, height) in pixels
    return out, framed_sizes


def place_view(image, mask, box, k, size, k_x=None):
    """Cut the subject out with its mask, scale it by k (k_x across, if given) and centre it on a black
    size x size square."""
    y0, y1, x0, x1 = box
    new_h = max(1, min(size, int(round((y1 - y0) * k))))
    new_w = max(1, min(size, int(round((x1 - x0) * (k if k_x is None else k_x)))))
    alpha = mask[y0:y1, x0:x1].float().clamp(0.0, 1.0)[..., None]
    crop = torch.cat([image[y0:y1, x0:x1, :3].float() * alpha, alpha], dim=-1)  # premultiplied on black
    crop = comfy.utils.common_upscale(crop.movedim(-1, 0)[None], new_w, new_h, "lanczos", "disabled")[0]
    crop = crop.movedim(0, -1).clamp(0.0, 1.0)
    canvas = torch.zeros((size, size, 3), dtype=torch.float32, device=crop.device)
    top, left = (size - new_h) // 2, (size - new_w) // 2
    canvas[top:top + new_h, left:left + new_w] = crop[..., :3]
    return canvas


def frame_pole_view(image, mask, side_sizes, size):
    """Frame a top or bottom view (drawn nose up) at the scale of the side views, then turn it half a turn.

    side_sizes: {view name: (width, height)} of the framed side views, in pixels. The subject's length is the
    width of its left / right view and the height of its top / bottom view; its width across is the width of
    its front / back view and of its top / bottom view. The half turn puts the nose at the bottom of the
    picture, which is how ComfyUI's pole cameras see a subject whose front faces the first orbit camera."""
    box = mask_bbox(mask)
    profile = [side_sizes[n][0] for n in ("left", "right") if n in side_sizes]
    across = [side_sizes[n][0] for n in ("front", "back") if n in side_sizes]
    k_length = max(profile) / float(box[1] - box[0]) if profile else None
    k_width = max(across) / float(box[3] - box[2]) if across else None
    if k_length is None and k_width is None:
        raise ValueError("Asset3DFrameViews: top / bottom views need at least one side view to take their scale from")
    # With both a profile and a front / back view the two axes are fitted separately: a drawn or generated
    # plan view rarely has exactly the proportions the side views imply (measured: 1.62 : 1 against 1.37 : 1),
    # and the mesh follows the side views.
    k = k_length if k_length is not None else k_width
    return torch.flip(place_view(image, mask, box, k, size, k_x=k_width), dims=[0, 1])


class Asset3DFrameViews(IO.ComfyNode):
    @classmethod
    def define_schema(cls):
        return IO.Schema(
            node_id="Asset3DFrameViews",
            display_name="Asset3D Frame Views (one scale)",
            category="3d/asset3d",
            description=(
                "Cuts each reference view out with its mask and places it centred on a black square, with the "
                "subject at the SAME scale in every view, as the Pixal3D multi-view rig expects. Use instead of "
                "cropping each view to its own mask, which enlarges the narrow views of long subjects."
            ),
            inputs=[
                IO.Combo.Input("mode", options=FRAMING_MODES, default="match_height",
                               tooltip="match_height = scale every view so the subject is equally tall (robust "
                                       "to views drawn at different sizes). source_scale = keep the pictures' "
                                       "own relative scale. per_view = fill the frame in each view (stock "
                                       "behaviour; right only when height dominates every view)."),
                IO.Int.Input("size", default=1024, min=256, max=2048, step=8),
                IO.Float.Input("pad_factor", default=1.1, min=1.0, max=2.0, step=0.01,
                               tooltip="The subject's longest side spans 1/pad_factor of the frame. 1.1 is the "
                                       "Pixal3D rig."),
                *[IO.Image.Input(f"{name}_image", optional=True) for name in VIEW_NAMES],
                *[IO.Mask.Input(f"{name}_mask", optional=True) for name in VIEW_NAMES],
                *[IO.Image.Input(f"{name}_image", optional=True,
                                 tooltip=f"The {name} view, drawn with the subject's front pointing to the top of "
                                         "the picture. Used for texture projection only.") for name in POLE_NAMES],
                *[IO.Mask.Input(f"{name}_mask", optional=True) for name in POLE_NAMES],
            ],
            outputs=[IO.Image.Output(display_name=name) for name in VIEW_NAMES + POLE_NAMES],
        )

    @classmethod
    def execute(cls, mode, size, pad_factor, **kwargs) -> IO.NodeOutput:
        def view(name):
            image, mask = kwargs.get(f"{name}_image"), kwargs.get(f"{name}_mask")
            if image is None:
                return None
            if mask is None:
                raise ValueError(f"Asset3DFrameViews: {name}_image is connected without {name}_mask")
            image, mask = image[0], mask[0]
            if mask.shape != image.shape[:2]:
                mask = torch.nn.functional.interpolate(mask[None, None].float(), size=image.shape[:2],
                                                       mode="bilinear")[0, 0]
            return image, mask

        sides = {name: view(name) for name in VIEW_NAMES}
        sides = {name: v for name, v in sides.items() if v is not None}
        if not sides:
            raise ValueError("Asset3DFrameViews needs at least one side view")
        framed_list, sizes = frame_views([v[0] for v in sides.values()], [v[1] for v in sides.values()],
                                         mode, size, pad_factor)
        framed = dict(zip(sides, framed_list))
        side_sizes = dict(zip(sides, sizes))
        for name in POLE_NAMES:
            pole = view(name)
            if pole is not None:
                framed[name] = frame_pole_view(pole[0], pole[1], side_sizes, size)
        out_device = comfy.model_management.intermediate_device()
        return IO.NodeOutput(*[framed[name][None].to(out_device) if name in framed else None
                               for name in VIEW_NAMES + POLE_NAMES])


# ---------------------------------------------------------------------------------------------------------
# UltraShape refine
# ---------------------------------------------------------------------------------------------------------

def list_checkpoints():
    found = []
    for folder in MODEL_FOLDERS:
        path = os.path.join(folder_paths.models_dir, folder)
        if os.path.isdir(path):
            found += [f for f in os.listdir(path) if f.lower().endswith(CHECKPOINT_EXTENSIONS)]
    return sorted(set(found)) or [DEFAULT_CHECKPOINT]


def find_checkpoint(name):
    for folder in MODEL_FOLDERS:
        path = os.path.join(folder_paths.models_dir, folder, name)
        if os.path.isfile(path):
            return path
    raise FileNotFoundError(
        f"UltraShape checkpoint '{name}' not found. Put it in ComfyUI/models/UltraShape/ "
        "(https://huggingface.co/infinith/UltraShape, ultrashape_v1.pt, 7.4 GB). See SETUP.md.")


def find_ultrashape_dir():
    """Folder that contains the `ultrashape` python package and `configs/infer_dit_refine.yaml`."""
    candidates = []
    env = os.environ.get("ASSET3D_ULTRASHAPE_DIR")
    if env:
        candidates.append(env)
    for base in folder_paths.get_folder_paths("custom_nodes"):
        if not os.path.isdir(base):
            continue
        for pack in sorted(os.listdir(base)):
            candidates.append(os.path.join(base, pack, "UltraShape-1.0"))  # jtydhr88/ComfyUI-UltraShape1
            candidates.append(os.path.join(base, pack, "ultrashape_repo"))
    for path in candidates:
        if os.path.isdir(os.path.join(path, "ultrashape")) and \
                os.path.isfile(os.path.join(path, "configs", CONFIG_NAME)):
            return path
    raise RuntimeError(
        "UltraShape is not installed. Install the node pack https://github.com/jtydhr88/ComfyUI-UltraShape1 "
        "(it carries the UltraShape-1.0 code) or set ASSET3D_ULTRASHAPE_DIR to an UltraShape-1.0 checkout. "
        "See SETUP.md. To run without refinement, set in_use_ultrashape to false.")


def free_vram():
    gc.collect()
    comfy.model_management.soft_empty_cache(force=True)


def yaw_matrix(degrees):
    a = math.radians(degrees)
    c, s = math.cos(a), math.sin(a)
    return np.array([[c, 0.0, s], [0.0, 1.0, 0.0], [-s, 0.0, c]], dtype=np.float64)


def drop_small_components(trimesh, vertices, faces, fraction):
    """Keep the connected pieces with at least `fraction` of the largest piece's face count."""
    adjacency = trimesh.graph.face_adjacency(faces=faces)
    labels = trimesh.graph.connected_component_labels(adjacency, node_count=len(faces))
    counts = np.bincount(labels)
    keep = counts >= max(1.0, fraction * counts.max())
    if keep.all():
        return vertices, faces
    faces = faces[keep[labels]]
    used = np.unique(faces)
    remap = np.full(len(vertices), -1, dtype=np.int64)
    remap[used] = np.arange(len(used))
    LOG.info("UltraShape refine: dropped %d small floating piece(s)", int((~keep).sum()))
    return vertices[used], remap[faces]


def mesh_item_to_numpy(mesh):
    """First item of a native MESH batch as (vertices [N,3] float64, faces [M,3] int64)."""
    vertices, faces = mesh.vertices, mesh.faces
    if getattr(mesh, "vertex_counts", None) is not None:
        vertices = vertices[0, :int(mesh.vertex_counts[0].item())]
        faces = faces[0, :int(mesh.face_counts[0].item())]
    else:
        if vertices.ndim == 3:
            vertices = vertices[0]
        if faces.ndim == 3:
            faces = faces[0]
    return (vertices.detach().float().cpu().numpy().astype(np.float64),
            faces.detach().cpu().numpy().astype(np.int64))


def image_to_rgba(image, mask):
    """IMAGE [B,H,W,3|4] + optional foreground MASK [B,H,W] -> PIL RGBA of the first item."""
    from PIL import Image
    img = image[0].detach().float().cpu().clamp(0.0, 1.0)
    rgb = img[..., :3]
    if mask is not None:
        alpha = mask[0].detach().float().cpu().clamp(0.0, 1.0)
        if alpha.shape != rgb.shape[:2]:
            alpha = torch.nn.functional.interpolate(alpha[None, None], size=rgb.shape[:2], mode="bilinear")[0, 0]
    elif img.shape[-1] == 4:
        alpha = img[..., 3]
    else:
        LOG.warning("Asset3DUltraShapeRefine: no mask connected, treating near-black pixels as background")
        alpha = (rgb.max(dim=-1).values > 0.02).float()
    alpha = torch.where(alpha < 0.1, torch.zeros_like(alpha), alpha)  # faint matte noise would break recentering
    if float(alpha.max()) <= 0.0:
        raise ValueError("Asset3DUltraShapeRefine: the mask is empty, there is no subject in the image")
    rgba = torch.cat([rgb, alpha[..., None]], dim=-1)
    return Image.fromarray((rgba.numpy() * 255.0).round().astype(np.uint8), mode="RGBA")


# The checkpoint stores its DINOv2 image encoder under the layer names of transformers 4.x. transformers 5
# renamed the attention projections; the tensors themselves are unchanged.
DINOV2_RENAMES = (
    (".attention.attention.query.", ".attention.q_proj."),
    (".attention.attention.key.", ".attention.k_proj."),
    (".attention.attention.value.", ".attention.v_proj."),
    (".attention.output.dense.", ".attention.o_proj."),
)


def rename_for_model(state_dict, model):
    """Rename checkpoint keys the installed transformers no longer uses to the names `model` expects."""
    expected = set(model.state_dict())
    out = {}
    for key, value in state_dict.items():
        if key not in expected:
            for old, new in DINOV2_RENAMES:
                if old in key and key.replace(old, new) in expected:
                    key = key.replace(old, new)
                    break
        out[key] = value
    return out


def build_pipeline(ultrashape_dir, checkpoint_path, device, dtype, low_vram):
    from omegaconf import OmegaConf
    from ultrashape.pipelines import UltraShapePipeline
    from ultrashape.utils.misc import instantiate_from_config

    cfg = OmegaConf.load(os.path.join(ultrashape_dir, "configs", CONFIG_NAME))
    params = cfg.model.params
    vae = instantiate_from_config(params.vae_config)
    dit = instantiate_from_config(params.dit_cfg)
    conditioner = instantiate_from_config(params.conditioner_config)
    scheduler = instantiate_from_config(params.scheduler_cfg)
    image_processor = instantiate_from_config(params.image_processor_cfg)

    if checkpoint_path.lower().endswith(".safetensors"):
        raise ValueError("Asset3DUltraShapeRefine expects the original ultrashape_v1.pt checkpoint")
    weights = torch.load(checkpoint_path, map_location="cpu", weights_only=True, mmap=True)
    vae.load_state_dict(weights["vae"], strict=True)
    dit.load_state_dict(weights["dit"], strict=True)
    conditioner.load_state_dict(rename_for_model(weights["conditioner"], conditioner), strict=True)
    del weights
    gc.collect()

    vae.eval().to(device, dtype=dtype)
    dit.eval().to(device, dtype=dtype)
    conditioner.eval().to(device, dtype=dtype)
    if hasattr(vae, "enable_flashvdm_decoder"):
        vae.enable_flashvdm_decoder()

    pipeline = UltraShapePipeline(vae=vae, model=dit, scheduler=scheduler, conditioner=conditioner,
                                  image_processor=image_processor)
    if low_vram:
        pipeline.enable_model_cpu_offload()
    return pipeline, int(params.vae_config.params.voxel_query_res)


class Asset3DUltraShapeRefine(IO.ComfyNode):
    @classmethod
    def define_schema(cls):
        return IO.Schema(
            node_id="Asset3DUltraShapeRefine",
            display_name="Asset3D UltraShape Refine",
            category="3d/asset3d",
            description=(
                "Refines the geometry of a coarse mesh with UltraShape 1.0, guided by one reference image. "
                "Returns an untextured high-poly MESH in the same position, scale and orientation as the input, "
                "so the generator's voxel colours can still be baked onto it. Frees ComfyUI's loaded models "
                "before it starts and its own model when it ends."
            ),
            is_experimental=True,
            inputs=[
                IO.Mesh.Input("mesh", tooltip="Coarse mesh. Decimate it to about 500k faces first."),
                IO.Image.Input("image", tooltip="The front view, cropped to the subject (same framing as the "
                                                "generator saw)."),
                IO.Mask.Input("mask", optional=True, tooltip="Foreground mask of that image (white = subject)."),
                IO.Combo.Input("checkpoint", options=list_checkpoints()),
                IO.Int.Input("steps", default=50, min=10, max=200, step=1,
                             tooltip="50 is the authors' setting; 12 works for a quick look."),
                IO.Float.Input("guidance_scale", default=5.0, min=1.0, max=15.0, step=0.5),
                IO.Int.Input("octree_resolution", default=384, min=256, max=1024, step=64,
                             tooltip="Detail and face count of the result, and the VRAM of the decode. "
                                     "384 is the safe start on 24 GB; try 512 next. 1024 ran out of memory on "
                                     "24 GB cards for several users."),
                IO.Int.Input("num_latents", default=16384, min=4096, max=32768, step=1024,
                             tooltip="Latent tokens. 32768 is the authors' setting and needs the most VRAM."),
                IO.Int.Input("num_chunks", default=8000, min=1000, max=50000, step=1000,
                             tooltip="Decode chunk size. Lower = less VRAM, slower."),
                IO.Int.Input("refine_seed", default=42, min=0, max=0x7FFFFFFF),
                IO.Combo.Input("dtype", options=["bfloat16", "float16", "float32"], default="bfloat16"),
                IO.Boolean.Input("low_vram", default=True,
                                 tooltip="Keep only the active sub-model on the GPU (slower, much safer)."),
                IO.Float.Input("normalize_scale", default=0.99, min=0.5, max=1.0, step=0.01, advanced=True,
                               tooltip="How much of UltraShape's unit cube the mesh fills. Lower to 0.95 on an "
                                       "'index out of bounds' error."),
                IO.Combo.Input("yaw_degrees", options=["0", "90", "180", "270"], default="0", advanced=True,
                               tooltip="Turn the mesh about the up axis before refining, if the result looks "
                                       "refined against the wrong side of the image. Undone on output."),
                IO.Combo.Input("align", options=["undo_normalize", "match_bounds"], default="undo_normalize",
                               advanced=True,
                               tooltip="How the result is put back in the input's frame. undo_normalize inverts "
                                       "UltraShape's centering and scaling exactly. match_bounds fits the "
                                       "result's bounding box to the input's."),
                IO.Boolean.Input("flip_faces", default=False, advanced=True,
                                 tooltip="Reverse the triangle winding if the result renders inside out."),
                IO.Float.Input("drop_floaters", default=0.01, min=0.0, max=0.5, step=0.005, advanced=True,
                               tooltip="Remove disconnected pieces smaller than this fraction of the largest "
                                       "piece's faces. UltraShape leaves small floating bits. 0 = keep all."),
            ],
            outputs=[IO.Mesh.Output(display_name="mesh")],
        )

    @classmethod
    def execute(cls, mesh, image, checkpoint, steps, guidance_scale, octree_resolution, num_latents, num_chunks,
                refine_seed, dtype, low_vram, normalize_scale, yaw_degrees, align, flip_faces,
                drop_floaters=0.01, mask=None) -> IO.NodeOutput:
        checkpoint_path = find_checkpoint(checkpoint)
        ultrashape_dir = find_ultrashape_dir()
        if ultrashape_dir not in sys.path:
            sys.path.insert(0, ultrashape_dir)
        try:
            import trimesh
            from ultrashape.surface_loaders import SharpEdgeSurfaceLoader
            from ultrashape.utils import voxelize_from_point
        except ImportError as e:
            raise RuntimeError(
                f"UltraShape's python dependencies are missing ({e}). Install the requirements of "
                "ComfyUI-UltraShape1 into ComfyUI's python. See SETUP.md.") from e

        device = comfy.model_management.get_torch_device()
        if device.type != "cuda":
            raise RuntimeError("Asset3DUltraShapeRefine needs a CUDA GPU")
        torch_dtype = {"bfloat16": torch.bfloat16, "float16": torch.float16, "float32": torch.float32}[dtype]

        # Coarse mesh -> trimesh, in the frame UltraShape will see.
        vertices, faces = mesh_item_to_numpy(mesh)
        if faces.shape[0] == 0:
            raise ValueError("Asset3DUltraShapeRefine: the input mesh has no faces")
        rotation = yaw_matrix(float(yaw_degrees))
        vertices = vertices @ rotation.T
        lo, hi = vertices.min(axis=0), vertices.max(axis=0)
        center, extent = (lo + hi) / 2.0, float((hi - lo).max())
        coarse = trimesh.Trimesh(vertices=vertices, faces=faces, process=False)
        pil_image = image_to_rgba(image, mask)

        # UltraShape lives outside ComfyUI's model management: make room first.
        comfy.model_management.unload_all_models()
        free_vram()

        pipeline = None
        try:
            pipeline, voxel_res = build_pipeline(ultrashape_dir, checkpoint_path, device, torch_dtype, low_vram)

            loader = SharpEdgeSurfaceLoader(num_sharp_points=204800, num_uniform_points=204800)
            surface = loader(coarse, normalize_scale=normalize_scale).to(device, dtype=torch_dtype)
            _, voxel_idx = voxelize_from_point(surface[:, :, :3], num_latents, resolution=voxel_res)
            del surface

            pbar = comfy.utils.ProgressBar(steps)
            done = [0]

            def callback(step_idx, t, outputs):
                comfy.model_management.throw_exception_if_processing_interrupted()
                done[0] += 1
                pbar.update_absolute(done[0], steps)

            generator = torch.Generator(device=device).manual_seed(refine_seed)
            LOG.info("UltraShape refine: steps=%d octree=%d latents=%d low_vram=%s",
                     steps, octree_resolution, num_latents, low_vram)
            with torch.autocast(device_type="cuda", dtype=torch_dtype):
                result, _latents = pipeline(
                    image=pil_image, voxel_cond=voxel_idx, generator=generator, box_v=1.0, mc_level=0.0,
                    octree_resolution=octree_resolution, num_inference_steps=steps, num_chunks=num_chunks,
                    guidance_scale=guidance_scale, callback=callback, callback_steps=1,
                )
            del _latents, voxel_idx
        finally:
            if pipeline is not None:
                try:
                    pipeline.to("cpu")
                except Exception:  # offload hooks may refuse the move; dropping the references is what matters
                    pass
            pipeline = None
            free_vram()

        refined = result[0] if isinstance(result, (list, tuple)) else result
        if refined is None or len(refined.faces) == 0:
            raise RuntimeError("UltraShape returned an empty mesh. Try another seed or a lower normalize_scale.")
        out_vertices = np.asarray(refined.vertices, dtype=np.float64)
        out_faces = np.asarray(refined.faces, dtype=np.int64)
        if drop_floaters > 0.0:
            out_vertices, out_faces = drop_small_components(trimesh, out_vertices, out_faces, drop_floaters)

        # Back to the input's frame: UltraShape centred the bbox and scaled its longest side to 2 * scale.
        if align == "match_bounds":
            rlo, rhi = out_vertices.min(axis=0), out_vertices.max(axis=0)
            out_vertices = (out_vertices - (rlo + rhi) / 2.0) * (extent / float((rhi - rlo).max())) + center
        else:
            out_vertices = out_vertices * (extent / (2.0 * normalize_scale)) + center
        out_vertices = out_vertices @ rotation  # inverse of the yaw applied above
        if flip_faces:
            out_faces = out_faces[:, ::-1]

        LOG.info("UltraShape refine done: %d vertices, %d faces", out_vertices.shape[0], out_faces.shape[0])
        out_device = comfy.model_management.intermediate_device()
        return IO.NodeOutput(Types.MESH(
            vertices=torch.from_numpy(np.ascontiguousarray(out_vertices)).float()[None].to(out_device),
            faces=torch.from_numpy(np.ascontiguousarray(out_faces)).int()[None].to(out_device),
        ))


class Asset3DExtension(ComfyExtension):
    @override
    async def get_node_list(self) -> list[type[IO.ComfyNode]]:
        return [Asset3DFrameViews, Asset3DShapeImages, Asset3DUltraShapeRefine]


# ---------------------------------------------------------------------------------------------------------
# GET /asset3d/status: what a caller cannot see in /object_info. The refine node registers without UltraShape,
# which is only looked for when it runs; this says beforehand whether its code, packages and checkpoint are here.
# It imports nothing of UltraShape.
# ---------------------------------------------------------------------------------------------------------

ULTRASHAPE_PACKAGES = ("trimesh", "omegaconf", "diffusers", "accelerate", "skimage", "pytorch_lightning")


def asset3d_status():
    import importlib.util
    try:
        find_ultrashape_dir()
        code = True
    except Exception:
        code = False
    missing = []
    for name in ULTRASHAPE_PACKAGES:
        try:
            if importlib.util.find_spec(name) is None:
                missing.append(name)
        except Exception:
            missing.append(name)
    checkpoints = []
    for folder in MODEL_FOLDERS:
        path = os.path.join(folder_paths.models_dir, folder)
        if os.path.isdir(path):
            checkpoints += [f for f in os.listdir(path) if f.lower().endswith(CHECKPOINT_EXTENSIONS)]
    return {"version": 1, "ultrashape": {"code": code, "packages": missing, "checkpoints": sorted(set(checkpoints))}}


def register_status_route():
    try:
        from aiohttp import web
        from server import PromptServer
        instance = getattr(PromptServer, "instance", None)
        if instance is None:
            return

        @instance.routes.get("/asset3d/status")
        async def status(request):
            return web.json_response(asset3d_status())
    except Exception as e:  # a ComfyUI without the server (tests, a headless import): the nodes still load
        LOG.warning("asset3d: status route not registered: %s", e)


register_status_route()


async def comfy_entrypoint() -> Asset3DExtension:
    return Asset3DExtension()
