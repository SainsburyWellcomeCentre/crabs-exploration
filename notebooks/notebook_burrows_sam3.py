"""Run SAM3 image inference on burrow frames using bbox/point prompts.

Follows the official SAM3 image predictor example:
https://github.com/facebookresearch/sam3/blob/main/examples/sam3_image_predictor_example.ipynb
and https://github.com/facebookresearch/sam3#basic-usage

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
#   "ipympl",
#   "scikit-image",
#   "scipy",
#   "xarray",
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

from datetime import datetime
from pathlib import Path

import matplotlib.patches as patches
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import zarr
from PIL import Image
from sam3 import build_sam3_image_model
from sam3.model.box_ops import box_xywh_to_cxcywh
from sam3.model.sam3_image_processor import Sam3Processor
from sam3.visualization_utils import (
    normalize_bbox,
    # TODO: replace with equivalent numpy function
)
from scipy import ndimage as ndi
from skimage.filters import gaussian, threshold_otsu
from skimage.measure import label as sk_label
from skimage.measure import regionprops

# %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
# Input data
# - images_dir: directory of PNG frames
# - prompt_coords_dir: directory of per-video prompt CSVs

images_dir = "/home/sminano/swc/project_crabs/burrow_mean_image_slurm_3014447"
manual_prompts_csv = (
    "/home/sminano/swc/project_crabs/manual_prompt_points_20260528_163908.csv"
)

# Prediction params
TEXT_PROMPT = "animal burrow entrance"  # set to None to skip the text prompt
CONF_THRESHOLD = (
    0.3  # masks below this threshold are not scaled up to full res
)

# -------------------------
# Postprocessing of masks
# -------------------------
# TODO: change to pixels
MIN_MASK_AREA_PIXELS = 100  # 200? in crab bodylengths?
MAX_MASK_AREA_FRAC = 0.05
MIN_SOLIDITY = 0.95  # 0.85

# Output dir for masks
# TODO: add timestamp
OUTPUT_DIR = Path(
    "/home/sminano/swc/project_crabs/crabs-exploration/output_burrows_sam3"
)

# %%
# %matplotlib widget

# %%%%%%%%%%
# Helpers
class ImageArrayLazy:
    """A lazy array for images in a list."""

    def __init__(self, img_paths):
        self.img_paths = sorted(img_paths)
        # add image shape, assuming all have same as
        # first sample
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


def create_mask_zarr(
    zarr_store_path, zarr_array_shape, zarr_metadata_dict=None
):
    """Create a zarr store for ID-encoded masks and write metadata."""
    # Unpack shape
    n_images, image_h, image_w = zarr_array_shape[:3]

    # Initialise store
    # TODO: does this match OCTRON output?
    mask_zarr = zarr.open(
        zarr_store_path,
        mode="w",
        shape=(n_images, image_h, image_w),
        dtype="int16",
        fill_value=0,  # background
        chunks=(1, image_h, image_w),
    )

    # Add metadata to store if available
    if zarr_metadata_dict is not None:
        mask_zarr.attrs.update(zarr_metadata_dict)
    return mask_zarr


def add_point_prompts(processor, inference_state, points_xy, labels=True):
    """Add point prompts to the inference state.

    This function mirrors ``Sam3Processor.add_geometric_prompt`` but appends
    points as geometric prompts instead of a bounding box. It relies on SAM3
    internals (``_get_dummy_prompt`` and ``_forward_grounding``) as the
    processor does not expose a public point method.

    All points are appended and the grounding forward pass runs **once** at
    the end (rather than once per point). Re-running ``_forward_grounding``
    after every point recomputes full-resolution masks for all accumulated
    objects each time, which is wasteful and blows up GPU memory.

    Note: ``points_xy`` is an ``(N, 2)`` array of ``(x, y)`` pairs normalized
    to ``[0, 1]`` (a single ``(x, y)`` pair is also accepted). ``labels``
    sets each input point as positive (``True``, default) or negative
    (``False``); pass a scalar to apply the same label to all points, or an
    ``(N,)`` sequence for per-point labels.
    """
    # Throw error if image not passed thru backbone yet
    # (if passed, SAM3 stashes the backbone resulting features under the key
    # "backbone_out" in the state dict; we need an image to ground against).
    if "backbone_out" not in inference_state:
        raise ValueError("call processor.set_image before adding a prompt")

    # Check if a text prompt was passed
    if "language_features" not in inference_state["backbone_out"]:
        # if not: set text prompt to a dummy "visual" text so the
        # model relies only on the geometric prompt (this is the standard
        # way SAM3 signals this, see `add_geometric_prompt`)
        dummy_text_outputs = processor.model.backbone.forward_text(
            ["visual"], device=processor.device
        )
        inference_state["backbone_out"].update(dummy_text_outputs)

    # Ensure a geometric_prompt container exists to append the point into.
    # (we need to initialise it here because we append to it later)
    if "geometric_prompt" not in inference_state:
        inference_state["geometric_prompt"] = (
            processor.model._get_dummy_prompt()
            # returns an empty "Prompt" container with number of boxes = 0;
            # an empty Prompt container (zero boxes, batch=1) that subsequent
            # append_boxes/append_points calls grow into.
            #
            # Note: the SAM3 image processor also supports internally mask prompts
            # but this is not exposed in the public API. Masks as prompts are used
            # internally by its tracker/video stack, when the previous frame mask
            # is passed as prompt to the next frame.
        )

    # Add input point prompts with their labels to the inference state
    # NOTE: here batch=1, n_points = N (the number of points passed in)
    points_xy = np.atleast_2d(
        np.asarray(points_xy, dtype=np.float32)
    )  # (N, 2)
    n_points = points_xy.shape[0]

    point_coords = torch.tensor(
        points_xy,
        device=processor.device,
        dtype=torch.float32,
    ).view(n_points, 1, 2)  # (n_points, batch, 2)

    point_labels = torch.as_tensor(
        np.broadcast_to(labels, (n_points,)),
        device=processor.device,
        dtype=torch.bool,
    ).view(n_points, 1)  # (n_points, batch)

    inference_state["geometric_prompt"].append_points(
        point_coords,
        point_labels,
        # point_masks, # default is None, could be removed
    )

    # Ground once over all accumulated prompts.
    return processor._forward_grounding(inference_state)


# TODO: review
def postprocess_masks(
    masks,
    min_area,
    max_area,
    min_solidity,
    verbose=True,
):
    """Split disconnected masks, drop ones too large/small or not blob-like.

    ``masks`` is (N, H, W) boolean. Returns ``(kept, kept_obj_idx,
    drop_counts)`` where ``kept`` is a list of (H, W) boolean masks,
    ``kept_obj_idx`` is the index (into ``masks``) of the source object each
    kept mask came from (so per-object data like scores can be mapped onto
    the kept masks), and ``drop_counts`` is a dict counting how many
    connected components were dropped per reason.

    Note that ``kept_obj_idx`` is needed because the function splits masks
    into connected components, so a single SAM3 object can yield several
    kept masks (or none), and each must inherit the right score.
    """
    # min_area = min_area_pixels * image_area
    # max_area = max_area_frac * image_area
    kept = []
    kept_obj_idx = []
    drop_counts = {"area_low": 0, "area_high": 0, "solidity": 0}
    for obj_idx, m in enumerate(masks.astype(bool)):
        # 1. split into connected components
        comp_labels = sk_label(m)
        for prop in regionprops(comp_labels):
            # 2. area filter
            if prop.area < min_area:
                drop_counts["area_low"] += 1
                if verbose:
                    print(
                        f"  obj {obj_idx} comp {prop.label}: dropped "
                        f"(area {prop.area} < {min_area:.0f})"
                    )
                continue
            if prop.area > max_area:
                drop_counts["area_high"] += 1
                if verbose:
                    print(
                        f"  obj {obj_idx} comp {prop.label}: dropped "
                        f"(area {prop.area} > {max_area:.0f})"
                    )
                continue
            # 3. blob-likeness
            if prop.solidity < min_solidity:
                drop_counts["solidity"] += 1
                if verbose:
                    print(
                        f"  obj {obj_idx} comp {prop.label}: dropped "
                        f"(solidity {prop.solidity:.2f} < {min_solidity})"
                    )
                continue
            kept.append(comp_labels == prop.label)
            kept_obj_idx.append(obj_idx)
    return kept, kept_obj_idx, drop_counts


# %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
# Load frames as a lazy array and map each frame to its video / date

list_image_files = sorted(list(Path(images_dir).glob("*.png")))
list_video_per_img = [
    img_p.stem.split("_", 1)[0].split("-Loop")[0] for img_p in list_image_files
]

# Get image array
image_array = ImageArrayLazy(list_image_files)
print(image_array.shape)


# %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
# Initialise the output ID-encoded mask zarr store

# Create a timestamped masks zarr store in the output directory
timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
output_masks_zarr = OUTPUT_DIR / f"masks_{timestamp}.zarr"

n_images, image_h, image_w = image_array.shape[:3]
metadata_dict = {
    "sam3_model": "sam3_image",
    "source_images_dir": str(images_dir),
    "manual_prompts_csv": str(manual_prompts_csv),
    "n_images": n_images,
    "image_shape": [image_h, image_w],
    "text_prompt": TEXT_PROMPT,
    "sam3_confidence_threshold": CONF_THRESHOLD,
    "postproc_min_mask_area_PIXELS": MIN_MASK_AREA_PIXELS,
    "postproc_max_mask_area_frac": MAX_MASK_AREA_FRAC,
    "postproc_min_solidity": MIN_SOLIDITY,
    "mask_encoding": "instance_id",
    "background_label": 0,
    "id_first_index": 1,
    # mask instance IDs in the zarr store start at 1 (not 0),
    # because 0 is reserved for the background label.
}
mask_zarr = create_mask_zarr(
    output_masks_zarr,
    (n_images, image_h, image_w),
    zarr_metadata_dict=metadata_dict,
)

# %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
# Load point prompts (one or more CSVs in prompt_coords_dir)
# CSV columns: group_id, prompt_point_x, prompt_point_y
# group_id has format "<video>_mean_n<frame>.png"; the video string is
# everything before "_mean".
df_prompts = pd.read_csv(manual_prompts_csv)

# point prompts in pixel xy, keyed by video string
points_xy_per_video = {
    video.split("_")[0]: group[["prompt_point_x", "prompt_point_y"]].to_numpy(
        dtype=np.float32
    )
    for video, group in df_prompts.groupby("group_id")
}


# %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
# Build SAM3 image model and processor

# Disable autograd for the whole notebook
torch.inference_mode().__enter__()

# to avoid bfloat16 (from model params) and float
# (from input) mismatch
if torch.cuda.is_available():
    torch.autocast("cuda", dtype=torch.bfloat16).__enter__()

# Instiantiate model: holds weights, submodules
# (backbone, text encoder, grounding head, mask decoder)
# and forward pass operations
model = build_sam3_image_model()

# instantiate the processor
# (a wrapper around the model, handling I/O conversion, building
# and mutation of the inference_state, prompt accumulation and
# confidence thresholding)
# NOTE: the model is accessible via processor.model
processor = Sam3Processor(model, confidence_threshold=CONF_THRESHOLD)


# %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
# Run inference on every frame and write ID-encoded masks to zarr

processed_frames = []
mask_scores_per_frame = {}  # frame_idx -> {mask_id: score}

for frame_idx in range(len(image_array)):
    # Load image
    image = Image.fromarray(image_array[frame_idx])
    width, height = image.size

    # Get corresponding video
    video_str = list_video_per_img[frame_idx]

    # Get normalised point prompts for that video 
    prompts_xy = points_xy_per_video.get(video_str)
    if prompts_xy is None or len(prompts_xy) == 0:
        print(f"Frame {frame_idx} ({video_str}): no prompts, skipping")
        continue
    # xy (pixels) -> normalized [0, 1]
    norm_points_xy = prompts_xy / np.array([width, height], dtype=np.float32)

    # --------------------
    # Pass image to processor and reset inference state
    inference_state = processor.set_image(image) # maybe: set_image_batch?
    processor.reset_all_prompts(
        inference_state
    )  # mutates the state dict in place

    # Add optional text prompt
    if TEXT_PROMPT is not None:
        inference_state = processor.set_text_prompt(
            state=inference_state, prompt=TEXT_PROMPT
        )

    # Add normalised point prompts (all at once; grounds a single time)
    inference_state = add_point_prompts(
        processor, inference_state, norm_points_xy, labels=True
    )
     # --------------------

    # Get predicted boolean masks and scores
    # Move masks to CPU, then release this frame's GPU state before the
    # next frame: otherwise the previous state (backbone features +
    # full-res masks_logits) stays alive during the next forward pass.
    masks = inference_state["masks"].cpu().numpy()
    scores = inference_state["scores"].float().cpu().numpy()  # (N,)
    del inference_state
    torch.cuda.empty_cache()

    masks = masks.squeeze(1) if masks.ndim == 4 else masks  # (N, H, W)
    n_objects = masks.shape[0]
    if n_objects == 0:
        print(f"Frame {frame_idx} ({video_str}): no detections")
        continue

    # -----------------------
    # Postprocess masks
    kept_masks, kept_obj_idx, drop_counts = postprocess_masks(
        masks,
        MIN_MASK_AREA_PIXELS,
        MAX_MASK_AREA_FRAC * image_h * image_w,
        MIN_SOLIDITY,
    )
    print(
        f"Frame {frame_idx} ({video_str}): postproc kept "
        f"{len(kept_masks)}/{n_objects} objects, "
        f"dropped {drop_counts}"
    )
    if not kept_masks:
        print(
            f"Frame {frame_idx} ({video_str}): no masks after postprocessing"
        )
        continue

    # -----------------------
    # Compute id-encoded mask
    # boolean masks (N, H, W) -> ID-encoded (H, W); higher ID wins on overlap
    #
    # n_masks may be different from n_objects because:
    # - SAM3 can return a mask that is all False
    # - when computing the id mask, if masks overlap we take the one with
    #   higher ID. So completely overlapping masks disappear.
    obj_ids = np.arange(1, len(kept_masks) + 1, dtype=np.int16)
    id_mask = np.zeros((image_h, image_w), dtype=np.int16)
    for oid, m in zip(obj_ids, kept_masks, strict=True):
        id_mask[m] = oid

    # Score per kept mask: each kept mask inherits the SAM3 score of the
    # source object it was split from (kept_obj_idx maps back into `scores`).
    kept_scores = scores[kept_obj_idx]
    id_to_score = {
        str(int(oid)): float(s)
        for oid, s in zip(obj_ids, kept_scores, strict=True)
    }
    # -----------------------

    mask_zarr[frame_idx] = id_mask
    processed_frames.append(frame_idx)
    mask_zarr.attrs["annotated_frames"] = processed_frames
    # per-mask scores, keyed by frame index then mask id (zarr attrs are
    # JSON, so keys are strings)
    mask_scores_per_frame[str(frame_idx)] = id_to_score
    mask_zarr.attrs["mask_scores"] = mask_scores_per_frame
    print(f"Frame {frame_idx} ({video_str}): {len(kept_masks)} masks")

print(f"Saved ID-encoded mask zarr to {output_masks_zarr}")

# %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
# Visualise ID-encoded masks in napari (masks over source frames)

import napari
import zarr

# Open the mask store (N, H, W) int16; 0 = background, >0 = instance ID.
masks = zarr.open(output_masks_zarr, mode="r")  # or pass a path string

# # Only some frames were annotated; the rest are all-zero. Restrict the
# # viewer to the annotated frames so you don't scroll through empty ones.
# annotated = masks.attrs.get("annotated_frames", list(range(masks.shape[0])))

# # Source frames as the background image (same H, W as the masks).
# frames = np.stack([image_array[i] for i in annotated])  # (n, H, W, C)
# labels = np.stack([masks[i] for i in annotated])          # (n, H, W)

viewer = napari.Viewer()
viewer.add_image(np.asarray(image_array), name="frames", rgb=True)
viewer.add_labels(masks, name="burrow masks")

# %%%%%%%%%%%%%%%%%
# Subsequent passes
# - pass the highest score ones that are not prompts?

# 1. Select masks that are higher than 0.5 score and do not overlap with a prompt point
# 2. Compute bounding box around those masks
# 3. Derive point prompt from the bbox using the existing function
# 4. Run inference with the old set of point prompts + new point prompts -- print how many new masks are found

# %%
# One manual iterative step.
# Reuses `kept_masks`, `kept_scores`, `points_xy` from the inference cell
# above (run for `select_frame_idx`, PROMPT_TYPE == "point").

PROMPT_SELECTION_MIN_SCORE = 0.0  # if CONF_THRESHOLD or 0, reuses all

img_iter = image_array[selected_frame_idx]
img_h_i, img_w_i = img_iter.shape[:2]

# 1. Select High-score kept masks that contain no existing prompt points
# get rows and column indices for each prompt
prompt_rc = np.round(prompts_xy[:, ::-1]).astype(int)  # (N, 2) as (row, col)
prompt_rows = prompt_rc[:, 0].clip(0, img_h_i - 1)
prompt_cols = prompt_rc[:, 1].clip(0, img_w_i - 1)

new_bboxes_xyxy = []
for m, s in zip(kept_masks, kept_scores):
    # skip masks with score below threshold
    if s <= PROMPT_SELECTION_MIN_SCORE:
        continue
    # skip masks already covered by a prompt point
    # (is any point prompt inside this mask?)
    if m[prompt_rows, prompt_cols].any():
        continue

    # if it passes previous checks:
    # compute bounding box around the mask -> pixel xyxy
    # output from np.where is row (y-axis), col (x-axis) coordinate of each
    # pixel in this mask
    ys, xs = np.where(m)
    # we get min/max to compute bbox
    new_bboxes_xyxy.append([xs.min(), ys.min(), xs.max(), ys.max()])

new_bboxes_xyxy = np.array(new_bboxes_xyxy, dtype=np.float32).reshape(-1, 4)
print(f"{len(new_bboxes_xyxy)} predicted masks selected for re-prompting")

# %%
# 3. Derive a point per new bbox 
new_points_xy = 0.5*(new_bboxes_xyxy[:,:2] + new_bboxes_xyxy[:,2:])

# %%
# plot old and new prompts
# old prompts (lime x), new prompts (cyan x) and the mask bboxes the new
# points were derived from (cyan rectangles).
fig, ax = plt.subplots()
ax.imshow(img_iter)
ax.scatter(
    prompts_xy[:, 0],
    prompts_xy[:, 1],
    c="lime",
    marker="x",
    s=120,
    label=f"old ({len(prompts_xy)})",
)
if len(new_points_xy):
    ax.scatter(
        new_points_xy[:, 0],
        new_points_xy[:, 1],
        c="cyan",
        marker="x",
        s=120,
        label=f"new ({len(new_points_xy)})",
    )
for x1, y1, x2, y2 in new_bboxes_xyxy:
    ax.add_patch(
        patches.Rectangle(
            (x1, y1),
            x2 - x1,
            y2 - y1,
            fill=False,
            linewidth=1.5,
            edgecolor="cyan",
        )
    )
ax.set_axis_off()
ax.legend(loc="upper right")
ax.set_title(
    f"{image_array.img_paths[selected_frame_idx].stem} - "
    f"{len(prompts_xy)} old + {len(new_points_xy)} new prompts"
)
plt.show()


# %%%%%%%%%%%%%%%%%
# Option C: tiled inference seeded by old + new point prompts
# -----------------------------------------------------------
# Exemplars are spatial, so they cannot be shared to a tile they do not
# fall inside. Here `points_xy` (old) and `new_points_xy` (new) from the
# iterative cell above form a single exemplar pool. The image is split into
# overlapping tiles; each tile runs SAM3 with the text prompt + whatever
# pool points land inside it (transformed to tile coords). Tiles with no
# exemplar are skipped (text-only finds nothing here -> residual gap).
# Per-tile masks are offset back to full-image coords and de-duplicated
# across tile seams by IoU.

TILE_SIZE = int(image_h / 3)  # tile side in pixels
TILE_OVERLAP = int(
    TILE_SIZE / 2
)  # int(image_w*0.05) #256    # overlap between neighbouring tiles, in pixels
MERGE_IOU = 0.5  # IoU above which two tile masks are the same burrow

img_full = image_array[selected_frame_idx]
H, W = img_full.shape[:2]
full_area = H * W

# exemplar pool: old + new point prompts (pixel xy, full-image frame)
pool_points = np.vstack([prompts_xy, new_points_xy]).astype(np.float32)
print(
    f"exemplar pool: {len(pool_points)} points "
    f"({len(prompts_xy)} old + {len(new_points_xy)} new)"
)


# %%
def _make_tiles(img_h, img_w, tile, overlap):
    """Overlapping (x0, y0, x1, y1) tiles covering the image."""
    step = tile - overlap
    tiles = set()
    for y0 in range(0, max(1, img_h - overlap), step):
        for x0 in range(0, max(1, img_w - overlap), step):
            # clamp to image, then shift back so the tile stays full-sized
            x1, y1 = min(x0 + tile, img_w), min(y0 + tile, img_h)
            tiles.add((max(0, x1 - tile), max(0, y1 - tile), x1, y1))
    return sorted(tiles)


def _run_sam3_points(image_pil, points_xy_px, area_for_postproc):
    """Run SAM3 (text + points) on a (cropped) image; masks in its frame."""
    w, h = image_pil.size
    state = processor.set_image(image_pil)
    processor.reset_all_prompts(state)

    # text prompt
    if TEXT_PROMPT is not None:
        state = processor.set_text_prompt(state=state, prompt=TEXT_PROMPT)

    # point prompts (all at once; grounds a single time)
    state = add_point_prompts(
        processor,
        state,
        points_xy_px / np.array([w, h], dtype=np.float32),
        labels=True,
    )

    masks = state["masks"].cpu().numpy()
    scores = state["scores"].float().cpu().numpy()
    del state
    torch.cuda.empty_cache()

    masks = masks.squeeze(1) if masks.ndim == 4 else masks
    if masks.shape[0] == 0:
        return [], np.empty(0, dtype=float)

    # area thresholds use the FULL image area so the absolute pixel limits
    # stay constant regardless of tile size
    kept, kept_idx, _ = postprocess_masks(
        masks,
        MIN_MASK_AREA_PIXELS,
        MAX_MASK_AREA_FRAC * img_h_i * img_w_i,
        MIN_SOLIDITY,
        verbose=False,
    )
    return kept, scores[kept_idx]


def _merge_by_iou(masks, scores, iou_thresh):
    """Greedy IoU de-dup of full-image masks; higher score wins on overlap."""
    kept_m, kept_s = [], []
    for i in np.argsort(scores)[::-1]:  # high score first
        m = masks[i]
        area = m.sum()
        is_dup = False
        for km in kept_m:
            inter = np.logical_and(m, km).sum()
            union = area + km.sum() - inter
            if union and inter / union > iou_thresh:
                is_dup = True
                break
        if not is_dup:
            kept_m.append(m)
            kept_s.append(scores[i])
    return kept_m, np.array(kept_s)


# %%
# --- run tiled inference ---------------------------------------------------
tiles = _make_tiles(H, W, TILE_SIZE, TILE_OVERLAP)
tile_masks, tile_scores = [], []
n_empty = 0
for x0, y0, x1, y1 in tiles:
    in_tile = (
        (pool_points[:, 0] >= x0)
        & (pool_points[:, 0] < x1)
        & (pool_points[:, 1] >= y0)
        & (pool_points[:, 1] < y1)
    )
    if not in_tile.any():
        n_empty += 1
        continue  # no exemplar -> text-only finds nothing, skip this tile
    crop = Image.fromarray(img_full[y0:y1, x0:x1])
    kept, ksc = _run_sam3_points(
        crop,
        pool_points[in_tile] - np.array([x0, y0], dtype=np.float32),
        full_area,
    )
    # offset each tile mask back into the full-image canvas
    for m in kept:
        full = np.zeros((H, W), dtype=bool)
        full[y0:y1, x0:x1] = m
        tile_masks.append(full)
    tile_scores.extend(ksc.tolist())

print(f"{len(tiles)} tiles, {n_empty} skipped (no exemplar)")
print(f"{len(tile_masks)} raw tile masks before merge")
# Q: I assume this is before merging tiles?

merged_masks, merged_scores = _merge_by_iou(
    tile_masks, np.array(tile_scores, dtype=float), MERGE_IOU
)

# TODO: review this count, I dont get it
print(
    f"{len(merged_masks)} masks after cross-tile merge "
    f"(pass-1 had {len(kept_masks)})"
)

# %%
# Persist tiled-pass masks to zarr and free intermediates before plotting.
# A separate store from the pass-1 mask_zarr so both passes coexist on disk.

# TODO: ideally we save to zarr as we go!
import gc  # noqa: E402

if "tiled_mask_zarr" not in globals():
    output_tiled_masks_zarr = OUTPUT_DIR / f"masks_tiled_{timestamp}.zarr"
    tiled_metadata_dict = {
        **metadata_dict,
        "pass": "tiled",
        "tile_size": TILE_SIZE,
        "tile_overlap": TILE_OVERLAP,
        "merge_iou": MERGE_IOU,
    }
    tiled_mask_zarr = create_mask_zarr(
        output_tiled_masks_zarr,
        (n_images, image_h, image_w),
        zarr_metadata_dict=tiled_metadata_dict,
    )

# Encode merged masks as ID-mask (higher score won the IoU merge, so order
# in `merged_masks` is high-to-low score; IDs follow that order).
tiled_id_mask = np.zeros((image_h, image_w), dtype=np.int16)
for oid, m in enumerate(merged_masks, start=1):
    tiled_id_mask[m] = oid
tiled_mask_zarr[selected_frame_idx] = tiled_id_mask

tiled_id_to_score = {str(i + 1): float(s) for i, s in enumerate(merged_scores)}
tiled_scores_attr = dict(tiled_mask_zarr.attrs.get("mask_scores", {}))
tiled_scores_attr[str(selected_frame_idx)] = tiled_id_to_score
tiled_mask_zarr.attrs["mask_scores"] = tiled_scores_attr
print(f"Saved tiled ID-mask zarr to {output_tiled_masks_zarr}")

# Drop the big per-tile mask lists now that the result is on disk.
del tile_masks, tile_scores, merged_masks, merged_scores
gc.collect()
torch.cuda.empty_cache()


# %%
# --- plot: pass-1 vs tiled result -----------------------------------------
# Read both ID-masks back from zarr so we don't hold N full-res bool masks
# in RAM, and use imshow instead of N ax.contour calls (much lighter).
pass1_id_mask = mask_zarr[selected_frame_idx]
tiled_id_mask = tiled_mask_zarr[selected_frame_idx]
n_pass1 = int(pass1_id_mask.max())
n_tiled = int(tiled_id_mask.max())

fig, axes = plt.subplots(1, 2, figsize=(16, 8))
for ax, (title, id_arr) in zip(
    axes,
    [
        (f"pass 1 - {n_pass1} masks", pass1_id_mask),
        (f"tiled (option C) - {n_tiled} masks", tiled_id_mask),
    ],
    strict=True,
):
    ax.imshow(img_full)
    ax.imshow(
        np.ma.masked_where(id_arr == 0, id_arr),
        cmap="tab10",
        alpha=0.5,
        interpolation="nearest",
    )
    ax.set_axis_off()
    ax.set_title(title)
# tile boundaries + exemplar pool overlaid on the tiled result
for x0, y0, x1, y1 in tiles:
    axes[1].add_patch(
        patches.Rectangle(
            (x0, y0),
            x1 - x0,
            y1 - y0,
            fill=False,
            edgecolor="cyan",
            linewidth=0.7,
            linestyle=":",
        )
    )
axes[1].scatter(
    pool_points[:, 0], pool_points[:, 1], c="lime", marker="x", s=40
)
plt.tight_layout()
plt.show()
# %%
# %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
# Plot trajectories on top of tiled burrow masks
# Overlay per-individual crab trajectories (from the CrabTracks zarr datatree)
# on the selected frame, alongside the tiled-pass burrow masks.

import xarray as xr  # noqa: E402

crabs_zarr_dataset = (
    Path.home()
    / "swc"
    / "project_crabs"
    / "data"
    / "_CrabTracks"
    / "CrabTracks-slurm2478780-2478861-2489356.zarr"  # "CrabTracks-slurm3012633.zarr"
)

dt = xr.open_datatree(crabs_zarr_dataset, engine="zarr", chunks={})

# Get video data
video_str = list_video_per_img[selected_frame_idx]
ds_video = dt[video_str].to_dataset()

# Flatten all (clip_id, time, individuals) samples and drop NaNs. No
# trajectory-length filter here — every non-NaN sample is plotted, and the
# minimum-hit threshold (MIN_HITS_PER_BURROW, below) controls which burrows
# get flagged in red.
position = ds_video.position  # (clip_id, time, individuals, space)
x = position.sel(space="x").values.reshape(-1)
y = position.sel(space="y").values.reshape(-1)
valid = ~np.isnan(x) & ~np.isnan(y)
x_clean, y_clean = x[valid], y[valid]
del x, y, valid

# Rasterise trajectory points with datashader: positions are
# (clip_id, time, individuals, space) padded to max time across clips,
# so we flatten x/y and drop NaNs (padding + missing detections) before
# aggregating onto a full-res canvas.
import datashader as ds  # noqa: E402
import datashader.transfer_functions as tf  # noqa: E402

DYNSPREAD_THRESHOLD = 0.975

img_h_i, img_w_i = image_array.img_h, image_array.img_w
canvas = ds.Canvas(
    plot_width=img_w_i,
    plot_height=img_h_i,
    x_range=(0, img_w_i),
    y_range=(0, img_h_i),
)
agg = canvas.points(pd.DataFrame({"x": x_clean, "y": y_clean}), "x", "y")
traj_img = tf.shade(agg, cmap=["#c3ff1f"])
traj_img = tf.dynspread(traj_img, threshold=DYNSPREAD_THRESHOLD)
# datashader y-origin is bottom; flip to match image y-origin (top)
traj_rgba = np.array(traj_img.to_pil().transpose(Image.FLIP_TOP_BOTTOM))

# ---------------------
# Combine pass-1 + tiled ID-masks into one with non-colliding IDs. Tiled
# IDs are shifted by n_pass1; on overlap pass-1 wins (i.e. tiled is only
# written where pass-1 is background). All masks are rendered with one
# colourmap so the two passes are visually undifferentiated.
pass1_id_mask = mask_zarr[selected_frame_idx]
tiled_id_mask = tiled_mask_zarr[selected_frame_idx]
n_pass1 = int(pass1_id_mask.max())
n_tiled = int(tiled_id_mask.max())

tiled_shifted = np.where(tiled_id_mask > 0, tiled_id_mask + n_pass1, 0)
combined_id_mask = np.where(
    pass1_id_mask > 0, pass1_id_mask, tiled_shifted
).astype(np.int32)
combined_masked = np.ma.masked_where(combined_id_mask == 0, combined_id_mask)
n_combined_max_id = int(combined_id_mask.max())

# Per-burrow trajectory-sample counts over the combined ID-mask. Sample
# the ID-mask at the integer pixel under each trajectory point, then
# bincount (index 0 = background, dropped via the [1:] slice below).
MIN_HITS_PER_BURROW = 30  # min trajectory samples inside a mask to flag it

traj_cols = np.round(x_clean).astype(int).clip(0, img_w_i - 1)
traj_rows = np.round(y_clean).astype(int).clip(0, img_h_i - 1)
ids_at_traj = combined_id_mask[traj_rows, traj_cols]
hits_per_id = np.bincount(ids_at_traj, minlength=n_combined_max_id + 1)
# +1 because IDs are 1-indexed (id 0 = background)
hit_ids = np.where(hits_per_id[1:] >= MIN_HITS_PER_BURROW)[0] + 1

# Interactive Plotly figure: saves a self-contained HTML and opens it in the
# browser. Legend has two toggles: one for "masks" (all mask fills + red
# hit-mask contours toggle together via legendgroup) and one for
# "trajectories" (the datashader-rasterised trajectory overlay).
import plotly.graph_objects as go  # noqa: E402
import plotly.io as pio  # noqa: E402
from skimage.measure import find_contours  # noqa: E402

pio.renderers.default = "browser"

# Build an RGBA overlay for the combined ID-mask using the tab10 colormap, so
# all masks render as a single go.Image trace (one legend entry).
tab10_rgb = (np.array(plt.cm.tab10.colors) * 255).astype(np.uint8)  # (10, 3)
mask_rgba = np.zeros((img_h_i, img_w_i, 4), dtype=np.uint8)
mask_nonzero = combined_id_mask > 0
mask_rgba[mask_nonzero, :3] = tab10_rgb[
    (combined_id_mask[mask_nonzero] - 1) % 10
]
mask_rgba[mask_nonzero, 3] = 128  # fill alpha ~= 0.5

fig = go.Figure()

# Background frame: layout image (sits below all traces, not toggleable)
fig.add_layout_image(
    source=Image.fromarray(image_array[selected_frame_idx]),
    xref="x",
    yref="y",
    x=0,
    y=0,
    sizex=img_w_i,
    sizey=img_h_i,
    sizing="stretch",
    layer="below",
)

# Masks overlay. go.Image traces cannot appear in the legend at all (no
# `showlegend`/`legendgroup` properties), so we toggle them via layout
# `updatemenus` buttons defined below. The red hit-mask contours are kept
# as always-visible Scatter traces (showlegend=False -> no legend clutter).
masks_trace_idx = len(fig.data)
fig.add_trace(
    go.Image(
        z=mask_rgba,
        colormodel="rgba256",
        name=f"masks ({n_pass1} pass-1 + {n_tiled} tiled)",
        hoverinfo="skip",
    )
)

# Red contours for hit masks: always visible, no legend entries
for mid in hit_ids:
    for contour in find_contours(combined_id_mask == mid, 0.5):
        fig.add_trace(
            go.Scatter(
                x=contour[:, 1],
                y=contour[:, 0],
                mode="lines",
                line=dict(color="red", width=1.5),
                showlegend=False,
                hoverinfo="skip",
            )
        )

# Trajectories overlay.
# colormodel="rgba256" is required so plotly honours the alpha channel of
# `z`; the default "rgb" would render the trajectory image as fully opaque
# lime and hide every layer beneath it.
traj_trace_idx = len(fig.data)
fig.add_trace(
    go.Image(
        z=traj_rgba,
        colormodel="rgba256",
        name=f"trajectories ({position.sizes['individuals']} individuals)",
        hoverinfo="skip",
    )
)

# Manual prompts as crosses
manual_pts = points_xy_per_video[video_str]
prompts_trace_idx = len(fig.data)
fig.add_trace(
    go.Scatter(
        x=manual_pts[:, 0],
        y=manual_pts[:, 1],
        mode="markers",
        marker=dict(symbol="x", color="lime", size=10, line=dict(width=0.1)),
        name=f"manual prompts ({len(manual_pts)})",
        showlegend=False,
        hoverinfo="skip",
    )
)

fig.update_layout(
    title=(
        f"{image_array.img_paths[selected_frame_idx].stem} - "
        f"{position.sizes['individuals']} trajectories, "
        f"{n_pass1} pass-1 + {n_tiled} tiled masks "
        f"({len(hit_ids)} with >= {MIN_HITS_PER_BURROW} samples)"
    ),
    xaxis_title="x (pixels)",
    yaxis_title="y (pixels)",
    yaxis_scaleanchor="x",
    plot_bgcolor="white",
    paper_bgcolor="white",
    showlegend=False,
    # Toggle buttons for the two image overlays. Each button uses
    # args/args2 so it acts as an on/off switch (click once -> args
    # applied, click again -> args2 applied).
    updatemenus=[
        dict(
            type="buttons",
            direction="right",
            x=0,
            xanchor="left",
            y=1.08,
            yanchor="bottom",
            showactive=False,
            buttons=[
                dict(
                    label="toggle masks",
                    method="restyle",
                    args=[{"visible": False}, [masks_trace_idx]],
                    args2=[{"visible": True}, [masks_trace_idx]],
                ),
                dict(
                    label="toggle trajectories",
                    method="restyle",
                    args=[{"visible": False}, [traj_trace_idx]],
                    args2=[{"visible": True}, [traj_trace_idx]],
                ),
                dict(
                    label="toggle prompts",
                    method="restyle",
                    args=[{"visible": False}, [prompts_trace_idx]],
                    args2=[{"visible": True}, [prompts_trace_idx]],
                ),
            ],
        )
    ],
    xaxis=dict(
        range=[0, img_w_i],
        showgrid=False,
        zeroline=False,
        linecolor="black",
        mirror=True,
        ticks="outside",
    ),
    yaxis=dict(
        range=[img_h_i, 0],  # invert y so image origin is top-left
        showgrid=False,
        zeroline=False,
        linecolor="black",
        mirror=True,
        ticks="outside",
    ),
)

frame_stem = image_array.img_paths[selected_frame_idx].stem
output_html = (
    OUTPUT_DIR / f"trajectories_burrows_{frame_stem}_{timestamp}.html"
)
fig.write_html(str(output_html), include_plotlyjs=True)
print(f"Saved interactive plot to {output_html}")
fig.show(renderer="browser")

# %%
# del processor, model; gc.collect(); torch.cuda.empty_cache()
# %%
