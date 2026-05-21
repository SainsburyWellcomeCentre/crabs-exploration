"""Interactive SAM3 burrow segmentation in a napari viewer.

A prototyping notebook that turns the batch workflow of
``notebook_burrows_sam3.py`` into an interactive tool:

  * Load an array of images into napari (one Image layer, navigated with the
    dims slider).
  * Click point prompts directly on each image (one nD Points layer, where the
    first coordinate is the frame index).
  * Press "Run inference" to run SAM3 on the image currently in view, using all
    point prompts placed on that frame. Predicted masks appear in a Labels
    layer.
  * Tune the confidence threshold *post-hoc* with a slider: SAM3 runs once at a
    low "inference floor"; the slider just re-filters the cached masks, so it is
    instant and does not touch the GPU.
  * Refine by adding/removing points and pressing "Run" again.
  * Manually edit masks with napari's paintbrush; edits auto-save to the zarr
    store (on every paint stroke and whenever you navigate to another frame).

Reproducibility: the point prompts per frame, the confidence threshold per
frame, the image file list and the SAM3 commit are all written to the zarr
store metadata. ``load_prompts_from_zarr`` reloads the prompts from a store.

Why an "inference floor": setting the SAM3 ``confidence_threshold`` very low
makes the model return a huge number of masks and causes CUDA OOM. Instead we
run once at a moderate floor (default 0.2) and keep every mask above it with its
score; the post-hoc slider can only move *up* from the floor. To explore lower
confidences, lower the floor in the UI and re-run.
"""
# /// script
# requires-python = ">=3.11"
# dependencies = [
#   "Pillow",
#   "zarr",
#   "torch>=2.5.1",
#   "torchvision>=0.20.1",
#   "sam3 @ git+https://github.com/facebookresearch/sam3.git",
#   "einops",
#   "huggingface_hub",
#   "scikit-image",
#   "scipy",
#   "napari[all]",
#   "magicgui",
# ]
#
# [tool.uv.sources]
# torch = { index = "pytorch-cu128" }
# torchvision = { index = "pytorch-cu128" }
#
# [[tool.uv.index]]
# name = "pytorch-cu128"
# url = "https://download.pytorch.org/whl/cu128"
# explicit = true
# ///

# %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
# Imports
import os

# Reduce CUDA allocator fragmentation. Must be set before torch is
# imported -> restart the kernel for this to take effect in a notebook.
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

import importlib.metadata
import json
import logging
from datetime import datetime
from pathlib import Path

import napari
import numpy as np
import torch
import zarr
from magicgui.widgets import (
    Container,
    FloatSlider,
    FloatSpinBox,
    LineEdit,
    PushButton,
)
from PIL import Image
from sam3 import build_sam3_image_model
from sam3.model.sam3_image_processor import Sam3Processor
from scipy import ndimage as ndi

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
logger = logging.getLogger("napari_sam3")


# %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
# Input data and defaults

# Directory of PNG frames to load into the viewer.
images_dir = "/home/sminano/swc/project_crabs/burrow_mean_image_slurm_3014447"

# Output dir for the masks zarr store.
OUTPUT_DIR = Path(
    "/home/sminano/swc/project_crabs/crabs-exploration/output_burrows_sam3"
)

# Defaults for the UI controls (all editable in the widget once it is open).
DEFAULT_TEXT_PROMPT = "burrow"  # empty string -> geometric-only inference
DEFAULT_INFERENCE_FLOOR = 0.2  # SAM3 confidence_threshold used at inference
DEFAULT_MAX_MASK_FRAC = 0.10  # drop masks larger than this fraction of pixels

# TODO: change to number of pixels
DEFAULT_MIN_MASK_FRAC = (
    0.000001  # drop masks smaller than this fraction of pixels
)


# %%%%%%%%%%
# Helpers


class ImageArrayLazy:
    """A lazy array for images in a list."""

    def __init__(self, img_paths):
        self.img_paths = sorted(img_paths)
        # add image shape, assuming all have same as first sample
        sample = np.array(Image.open(img_paths[0]))  # H, W, C
        self.img_h, self.img_w, self.img_c = sample.shape

    def __len__(self):
        return len(self.img_paths)

    def __getitem__(self, idx):
        return np.array(Image.open(self.img_paths[idx]))

    @property
    def shape(self):
        return (len(self.img_paths), self.img_h, self.img_w, self.img_c)
        # B, H, W, C


def create_mask_zarr(path_to_zarr, zarr_array_shape, metadata_dict=None):
    """Create a zarr store for ID-encoded masks and write metadata."""
    n_images, image_h, image_w = zarr_array_shape[:3]
    mask_zarr = zarr.open(
        path_to_zarr,
        mode="w",
        shape=(n_images, image_h, image_w),
        dtype="int16",
        fill_value=0,
        chunks=(1, image_h, image_w),
    )
    if metadata_dict is not None:
        mask_zarr.attrs.update(metadata_dict)
    return mask_zarr


def add_point_prompt(processor, state, point_xy, label=True):
    """Add a single point prompt and run inference, returning the updated state.

    ``point_xy`` is an ``(x, y)`` pair normalized to ``[0, 1]``. Mirrors
    ``Sam3Processor.add_geometric_prompt`` but appends a point to the geometric
    prompt instead of a box. Relies on SAM3 internals (``_get_dummy_prompt`` /
    ``_forward_grounding``) as the processor exposes no public point method.
    """
    if "backbone_out" not in state:
        raise ValueError("call processor.set_image before adding a prompt")
    if "language_features" not in state["backbone_out"]:
        # no text prompt yet: fall back to a dummy "visual" text prompt so the
        # model relies only on the geometric prompt
        dummy_text = processor.model.backbone.forward_text(
            ["visual"], device=processor.device
        )
        state["backbone_out"].update(dummy_text)
    if "geometric_prompt" not in state:
        state["geometric_prompt"] = processor.model._get_dummy_prompt()

    # points: (n_points, batch, 2); labels: (n_points, batch); mask: (batch, n)
    pts = torch.tensor(
        point_xy, device=processor.device, dtype=torch.float32
    ).view(1, 1, 2)
    lbl = torch.tensor(
        [label], device=processor.device, dtype=torch.bool
    ).view(1, 1)
    msk = torch.zeros(1, 1, dtype=torch.bool, device=processor.device)
    state["geometric_prompt"].append_points(pts, lbl, msk)

    return processor._forward_grounding(state)


def get_sam3_commit():
    """Best-effort SAM3 commit hash (for reproducibility metadata)."""
    try:
        dist = importlib.metadata.distribution("sam3")
        direct_url = dist.read_text("direct_url.json")
        if direct_url:
            info = json.loads(direct_url)
            vcs = info.get("vcs_info", {})
            return vcs.get("commit_id") or info.get("url", "unknown")
    except Exception:  # noqa: BLE001 - metadata is optional
        pass
    try:
        return importlib.metadata.version("sam3")
    except Exception:  # noqa: BLE001
        return "unknown"


# %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
# Postprocessing


def postprocess(id_mask, frame_idx, max_mask_frac, min_mask_frac):
    """Clean an ID-encoded mask; returns the cleaned mask and logs warnings.

    Order matters: disconnected masks are split first, so a small fragment
    split off a larger mask is still size-checked afterwards.

    - Disconnected masks: a mask whose pixels form several separate blobs is
      split, each connected blob becoming its own fresh ID.
    - Oversized masks: an ID covering more than ``max_mask_frac`` of the image
      is dropped.
    - Undersized masks: an ID covering less than ``min_mask_frac`` is dropped.
    - Overlap: already resolved upstream by the max-ID encoding.
    """
    total_px = id_mask.shape[0] * id_mask.shape[1]
    result = np.zeros_like(id_mask)
    next_id = 1

    for orig_id in np.unique(id_mask):
        if orig_id == 0:
            continue
        binary = id_mask == orig_id

        # number each connected blob of this ID separately
        labelled, n_blobs = ndi.label(binary)
        new_ids = []
        for blob in range(1, n_blobs + 1):
            blob_mask = labelled == blob
            frac = blob_mask.sum() / total_px
            if frac > max_mask_frac:
                logger.warning(
                    "frame %d, ID %d: dropped oversized mask "
                    "(%.1f%% of image > %.1f%%)",
                    frame_idx,
                    orig_id,
                    frac * 100,
                    max_mask_frac * 100,
                )
                continue
            if frac < min_mask_frac:
                logger.warning(
                    "frame %d, ID %d: dropped undersized mask "
                    "(%.3f%% of image < %.3f%%)",
                    frame_idx,
                    orig_id,
                    frac * 100,
                    min_mask_frac * 100,
                )
                continue
            result[blob_mask] = next_id
            new_ids.append(next_id)
            next_id += 1

        if n_blobs > 1 and new_ids:
            logger.warning(
                "frame %d, ID %d: mask was disconnected (%d blobs) -> "
                "split into IDs %s",
                frame_idx,
                orig_id,
                n_blobs,
                new_ids,
            )

    return result


# %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
# Load frames as a lazy array

list_image_files = sorted(Path(images_dir).glob("*.png"))
image_array = ImageArrayLazy(list_image_files)
n_images, image_h, image_w = image_array.shape[:3]
print(f"Loaded {n_images} images of shape {(image_h, image_w)}")


# %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
# Build SAM3 image model

# avoid bfloat16 / float mismatch
if torch.cuda.is_available():
    torch.autocast("cuda", dtype=torch.bfloat16).__enter__()

model = build_sam3_image_model()
SAM3_COMMIT = get_sam3_commit()
print(f"SAM3 commit: {SAM3_COMMIT}")


# %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
# Initialise the output ID-encoded mask zarr store and the in-memory masks

timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
output_masks_zarr = OUTPUT_DIR / f"masks_napari_{timestamp}.zarr"

mask_zarr = create_mask_zarr(
    output_masks_zarr,
    (n_images, image_h, image_w),
    metadata_dict={
        "timestamp": timestamp,
        "sam3_model": "sam3_image",
        "sam3_commit": SAM3_COMMIT,
        "source_images_dir": str(images_dir),
        "image_shape": [image_h, image_w],
        "image_files": [str(p) for p in image_array.img_paths],
        "mask_encoding": "instance_id",
        "background_label": 0,
        "id_offset": 1,
    },
)
print(f"Created mask zarr store at {output_masks_zarr}")

# In-memory copy backing the Labels layer. Each frame slice is written to the
# zarr store on save (avoids dask-write quirks while painting).
masks_in_memory = np.zeros((n_images, image_h, image_w), dtype=np.int16)


# %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
# Per-frame state
# - inference_cache: raw SAM3 output per frame, so the threshold slider can
#   re-filter instantly without re-running the model.
# - point_prompts_per_frame, threshold_per_frame: for reproducibility metadata.

inference_cache: dict[int, dict] = {}
point_prompts_per_frame: dict[int, list] = {}
threshold_per_frame: dict[int, float] = {}
annotated_frames: set[int] = set()


# %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
# Build the napari viewer and layers

viewer = napari.Viewer()
viewer.add_image(np.asarray(image_array), name="images")
points_layer = viewer.add_points(
    np.empty((0, 3)),
    ndim=3,
    name="point prompts",
    face_color="red",
    border_color="white",
    size=20,
)
labels_layer = viewer.add_labels(masks_in_memory, name="masks")

# select the Points layer so the user can immediately click prompts
viewer.layers.selection.active = points_layer
points_layer.mode = "add"


# %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
# Core operations


def current_frame():
    """Frame index currently shown by the dims slider."""
    return int(viewer.dims.current_step[0])


def points_for_frame(frame_idx):
    """``(M, 2)`` array of (x, y) pixel prompts placed on ``frame_idx``."""
    data = points_layer.data
    if len(data) == 0:
        return np.empty((0, 2), dtype=np.float32)
    on_frame = np.rint(data[:, 0]).astype(int) == frame_idx
    yx = data[on_frame, 1:]
    return yx[:, ::-1].astype(np.float32)  # (y, x) -> (x, y)


def write_metadata():
    """Refresh the reproducibility metadata in the zarr store attrs."""
    mask_zarr.attrs.update(
        {
            "point_prompts_per_frame": {
                str(k): np.asarray(v).tolist()
                for k, v in point_prompts_per_frame.items()
            },
            "confidence_threshold_per_frame": {
                str(k): float(v) for k, v in threshold_per_frame.items()
            },
            "annotated_image_files": [
                str(image_array.img_paths[i]) for i in sorted(annotated_frames)
            ],
            "text_prompt": text_prompt_w.value,
            "inference_floor": float(inference_floor_w.value),
            "max_mask_frac": float(max_mask_frac_w.value),
            "min_mask_frac": float(min_mask_frac_w.value),
        }
    )


def save_frame(frame_idx):
    """Write one frame's mask slice to the zarr store and refresh metadata."""
    mask_zarr[frame_idx] = labels_layer.data[frame_idx]
    annotated_frames.add(frame_idx)
    write_metadata()


def apply_threshold(frame_idx, thr=None):
    """Re-filter the cached SAM3 masks for a frame at confidence ``thr``.

    Pure post-hoc operation: no model call. Rebuilds the frame's mask from the
    cached SAM3 output, runs postprocessing, updates the Labels layer and saves.
    NOTE: this discards any manual paint edits on the frame -- set the
    threshold first, then paint.
    """
    cache = inference_cache.get(frame_idx)
    if cache is None:
        return
    if thr is None:
        thr = threshold_w.value

    masks_bool = cache["masks_bool"]
    scores = cache["scores"]
    keep = scores >= thr
    kept = masks_bool[keep]

    if len(kept) == 0:
        id_mask = np.zeros((image_h, image_w), dtype=np.int16)
    else:
        obj_ids = np.arange(1, len(kept) + 1, dtype=np.int16)[:, None, None]
        id_mask = (kept.astype(np.int16) * obj_ids).max(axis=0)

    id_mask = postprocess(
        id_mask, frame_idx, max_mask_frac_w.value, min_mask_frac_w.value
    )

    labels_layer.data[frame_idx] = id_mask.astype(np.int16)
    labels_layer.refresh()
    threshold_per_frame[frame_idx] = float(thr)
    save_frame(frame_idx)
    logger.info(
        "frame %d: %d masks at threshold %.2f",
        frame_idx,
        int(id_mask.max()),
        thr,
    )


def run_inference():
    """Run SAM3 on the frame in view using its point prompts."""
    frame_idx = current_frame()
    points_xy = points_for_frame(frame_idx)
    if len(points_xy) == 0:
        logger.warning(
            "frame %d: no point prompts placed, skipping", frame_idx
        )
        return

    image = Image.fromarray(image_array[frame_idx])
    width, height = image.size

    # rebuild the processor so the inference floor from the UI is used
    processor = Sam3Processor(
        model, confidence_threshold=float(inference_floor_w.value)
    )
    state = processor.set_image(image)
    processor.reset_all_prompts(state)

    text_prompt = text_prompt_w.value.strip()
    if text_prompt:
        state = processor.set_text_prompt(state=state, prompt=text_prompt)

    # normalize (x, y) pixel prompts to [0, 1] and add them one by one
    norm_xy = points_xy / np.array([width, height], dtype=np.float32)
    for px, py in norm_xy:
        state = add_point_prompt(
            processor, state, (float(px), float(py)), label=True
        )

    # move masks/scores to CPU, then release this frame's GPU state
    masks = state["masks"].cpu().numpy()
    scores = state["scores"].cpu().numpy().reshape(-1)
    del state
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    masks = masks.squeeze(1) if masks.ndim == 4 else masks  # (N, H, W)
    inference_cache[frame_idx] = {
        "masks_bool": masks.astype(bool),
        "scores": scores,
    }
    point_prompts_per_frame[frame_idx] = points_xy.tolist()
    logger.info(
        "frame %d: SAM3 returned %d masks (floor %.2f, %d point prompts)",
        frame_idx,
        masks.shape[0],
        inference_floor_w.value,
        len(points_xy),
    )

    apply_threshold(frame_idx)


# %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
# Build the dock widget

text_prompt_w = LineEdit(value=DEFAULT_TEXT_PROMPT, label="text prompt")
inference_floor_w = FloatSpinBox(
    value=DEFAULT_INFERENCE_FLOOR,
    min=0.0,
    max=1.0,
    step=0.05,
    label="inference floor",
)
max_mask_frac_w = FloatSpinBox(
    value=DEFAULT_MAX_MASK_FRAC,
    min=0.0,
    max=1.0,
    step=0.01,
    label="max mask frac",
)
min_mask_frac_w = FloatSpinBox(
    value=DEFAULT_MIN_MASK_FRAC,
    min=0.0,
    max=1.0,
    step=0.0005,
    label="min mask frac",
)
run_w = PushButton(text="Run inference")
threshold_w = FloatSlider(
    value=DEFAULT_INFERENCE_FLOOR,
    min=DEFAULT_INFERENCE_FLOOR,
    max=1.0,
    label="confidence threshold",
)


def _on_run():
    run_inference()


def _on_threshold(value):
    apply_threshold(current_frame(), thr=value)


def _on_floor_change(value):
    # the post-hoc slider can never go below the inference floor
    threshold_w.min = value
    if threshold_w.value < value:
        threshold_w.value = value


def _on_postproc_change(value):
    # re-run postprocessing on the current frame with the new size limits
    apply_threshold(current_frame())


run_w.clicked.connect(_on_run)
threshold_w.changed.connect(_on_threshold)
inference_floor_w.changed.connect(_on_floor_change)
max_mask_frac_w.changed.connect(_on_postproc_change)
min_mask_frac_w.changed.connect(_on_postproc_change)

widget = Container(
    widgets=[
        text_prompt_w,
        inference_floor_w,
        max_mask_frac_w,
        min_mask_frac_w,
        run_w,
        threshold_w,
    ]
)
viewer.window.add_dock_widget(widget, name="SAM3", area="right")


# %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
# Auto-save manual edits
# - on every paint stroke: save the frame in view
# - on navigating to another frame: save the frame we are leaving


def _on_paint(event):
    save_frame(current_frame())


_prev_frame = current_frame()


def _on_step_change(event):
    global _prev_frame
    new_frame = current_frame()
    if new_frame != _prev_frame:
        save_frame(_prev_frame)
        _prev_frame = new_frame


labels_layer.events.paint.connect(_on_paint)
viewer.dims.events.current_step.connect(_on_step_change)


# %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
# Run the napari event loop (only needed when running as a plain script)
napari.run()


# %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
# Reproducibility: reload point prompts from a saved zarr store
# Run this cell against an existing store to repopulate the Points layer and
# the per-frame thresholds, so an analysis can be reproduced from metadata.


def load_prompts_from_zarr(path, target_points_layer):
    """Repopulate a Points layer + threshold dict from a saved mask zarr.

    Returns the ``confidence_threshold_per_frame`` dict read from the store.
    """
    store = zarr.open(path, mode="r")
    prompts = store.attrs.get("point_prompts_per_frame", {})
    thresholds = store.attrs.get("confidence_threshold_per_frame", {})

    nd_points = []
    for frame_str, xy_list in prompts.items():
        frame = int(frame_str)
        for x, y in xy_list:  # stored as (x, y) pixels
            nd_points.append([frame, y, x])  # napari wants (frame, y, x)

    target_points_layer.data = (
        np.asarray(nd_points, dtype=np.float32)
        if nd_points
        else np.empty((0, 3), dtype=np.float32)
    )
    loaded_thresholds = {int(k): float(v) for k, v in thresholds.items()}
    print(f"Loaded prompts for {len(prompts)} frames from {path}")
    return loaded_thresholds


# Example (uncomment and point at an existing store):
# threshold_per_frame.update(
#     load_prompts_from_zarr(output_masks_zarr, points_layer)
# )
