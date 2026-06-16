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
from sam3.model.sam3_image_processor import Sam3Processor
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


# -------------------------
# Postprocessing of masks
# -------------------------
MIN_MASK_AREA_PIXELS = 200  # 100  # 200? in crab bodylengths?
MAX_MASK_AREA_PIXELS = 2500
MIN_SOLIDITY = 0.95

# ----------------------
# Output
# ---------------------
# Output dir for masks
# TODO: add timestamp
OUTPUT_DIR = Path(
    "/home/sminano/swc/project_crabs/crabs-exploration/output_burrows_sam3"
)

MAX_REGIONS_PER_IMAGE = 500

# -------------------------------------------------------------------
# Pass-2 input: masks zarr produced by notebook_burrows_sam3_pass_1.py
# (ID-encoded "masks" + per-region "scores"). Point this at the
# timestamped store written by the first pass.
# -------------------------------------------------------------------
PASS_1_MASKS_ZARR = OUTPUT_DIR / "masks_20260616_151414.zarr"

# Frame to run the second pass on
SELECTED_FRAME_IDX = 0

# Min score for a pass-1 mask to seed a new prompt in the second pass
PROMPT_SELECTION_MIN_SCORE = 0.0  # if CONF_THRESHOLD or 0, reuses all

# Inference params
TEXT_PROMPT = (
    "hole"  # "crab burrow in sand"  # set to None to skip the text prompt
)
CONF_THRESHOLD = (
    0.35  # masks below this threshold are not scaled up to full res
)


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


def initialise_mask_zarr(
    output_dir,
    images_dir,
    manual_prompts_csv,
    image_shape,
    text_prompt,
    conf_threshold,
    min_mask_area_pixels,
    max_mask_area_pixels,
    min_solidity,
    max_regions_per_image=500,
    data_pass=1,
):
    """Create mask ID encoded zarr store timestamped and with metadata."""
    # Create a timestamped masks zarr store in the output directory
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir.mkdir(parents=True, exist_ok=True)
    output_masks_zarr = output_dir / f"masks_pass_{data_pass}_{timestamp}.zarr"

    n_images, image_h, image_w = image_shape  # image_array.shape[:3]
    metadata_dict = {
        "sam3_model": "sam3_image",
        "source_images_dir": str(images_dir),
        "manual_prompts_csv": str(manual_prompts_csv),
        "n_images": n_images,
        "image_shape": [image_h, image_w],
        "estim_max_regions_per_image": max_regions_per_image,
        "text_prompt": text_prompt,
        "sam3_confidence_threshold": conf_threshold,
        "postproc_min_mask_area_PIXELS": min_mask_area_pixels,
        "postproc_max_mask_area_PIXELS": max_mask_area_pixels,
        "postproc_min_solidity": min_solidity,
        "mask_encoding": "instance_id",
        "background_label": 0,
        "id_first_index": 1,
        # mask instance IDs in the zarr store start at 1 (not 0),
        # because 0 is reserved for the background label.
    }

    # Initialise root
    root = zarr.open_group(output_masks_zarr, mode="w")

    # Add mask array
    _mask_zarr = root.create_array(
        "masks",
        shape=(n_images, image_h, image_w),
        dtype="int16",
        fill_value=0,  # background
        chunks=(1, image_h, image_w),
    )

    # Initialise scores array as a sibling of "masks"
    # shape (n_frames, max_regions + 1); column 0 = background, unused -> NaN
    _scores_zarr = root.create_array(
        "scores",
        shape=(n_images, max_regions_per_image + 1),
        chunks=(1, max_regions_per_image + 1),
        dtype="float32",
        fill_value=np.nan,
    )

    # Add metadata to root if available
    root.attrs.update(metadata_dict)

    return root, output_masks_zarr


def extract_normalised_point_prompts_per_video(
    manual_prompts_csv: Path | str, img_w: float, img_h: float
) -> dict:
    """Compute dict mapping video to normalised point prompt coordinates."""
    # Read from csv
    df_prompts = pd.read_csv(manual_prompts_csv)

    # point prompts in pixel xy ---> then normalised,
    # keyed by video string
    # normalise coordinates by image width and height
    points_xy_normalised_per_video = {
        video.split("_")[0]: (
            group[["prompt_point_x", "prompt_point_y"]].to_numpy()
            / np.array([img_w, img_h])
        ).astype(np.float32)
        for video, group in df_prompts.groupby("group_id")
    }

    return points_xy_normalised_per_video


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
        np.broadcast_to(labels, (n_points,)).copy(),
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


def postprocess_masks(
    masks: np.ndarray,
    scores: np.ndarray,
    min_area: int,
    max_area: int,
    min_solidity: float,
):
    """Split masks into connected regions and filter.

    Regions are filtered based on area range and solidity.

    ``masks`` is (N, H, W) boolean. Returns ``(kept, list_mask_idcs,
    drop_counts)`` where ``kept`` is a list of (H, W) boolean masks,
    ``list_mask_idcs`` is the index (into ``masks``) of the source object each
    kept mask came from (so per-object data like scores can be mapped onto
    the kept masks), and ``drop_counts`` is a dict counting how many
    connected regions were dropped per reason.

    Note that ``list_mask_idcs`` is needed because the function splits masks
    into connected components (regions), so a single SAM3 object can yield several
    kept masks (or none), and each must inherit the right score.
    """
    list_kept_bool_regions = []
    list_mask_idcs = []
    drop_counts = {"area_low": 0, "area_high": 0, "solidity": 0}

    # Loop thru masks
    for mask_idx, mask in enumerate(masks.astype(bool)):
        # Label connected regions in mask
        # (connected regions are assigned the same int)
        label_mask = sk_label(mask)

        # Compute properties per region
        list_regions_w_props = regionprops(label_mask)

        # Loop thru regions
        for region in list_regions_w_props:
            # Filter by min area
            if region.area < min_area:
                drop_counts["area_low"] += 1
                continue

            # Filter by max area
            if region.area > max_area:
                drop_counts["area_high"] += 1
                continue

            # Filter by solidity (proxy for blob-likeness)
            # (ratio of pixels in the region to pixels of the convex hull,
            # ranges from 0 (theoretical) to 1 (perfectly convex))
            # convex_area ≥ area ≥ 1 for any real region
            if region.solidity < min_solidity:
                drop_counts["solidity"] += 1
                continue

            # If all pass: retain that region within the mask
            list_kept_bool_regions.append(label_mask == region.label)
            # Keep track of the mask ID associated to this region too
            list_mask_idcs.append(mask_idx)

    # Get list of scores for kept regions
    list_kept_scores = scores[list_mask_idcs]

    return list_kept_bool_regions, list_kept_scores, drop_counts


def compute_new_prompts_from_masks(
    bool_masks: np.ndarray,
    scores: np.ndarray,
    prompts_xy_norm: np.ndarray,
    mask_min_score: float,
    *,
    img_h: int,
    img_w: int,
) -> np.ndarray:
    """Derive new normalised point prompts from mask bbox centroids.

    A mask seeds a new prompt only if its score is above ``mask_min_score``
    and none of the existing ``prompts_xy_norm`` fall inside it. The new
    prompt is the centroid of the mask's bounding box.

    ``bool_masks`` is (N, H, W) boolean and ``scores`` is (N,) aligned with
    it; ``prompts_xy_norm`` is (M, 2) normalised (x, y). Returns (K, 2)
    normalised (x, y) prompts (K <= N), or shape (0, 2) if none qualify.
    """
    # De-normalise existing prompts back to pixel (col, row), clipped
    prompt_cols, prompt_rows = (
        np.round(prompts_xy_norm * [img_w, img_h]).astype(int).T
    )
    prompt_rows = prompt_rows.clip(0, img_h - 1)
    prompt_cols = prompt_cols.clip(0, img_w - 1)

    centroids = []
    for mask, score in zip(bool_masks, scores, strict=True):
        # skip masks below the score threshold or already hit by a prompt
        if score <= mask_min_score or mask[prompt_rows, prompt_cols].any():
            continue

        # bbox centroid of the mask, in pixel (x, y)
        # (np.where returns row (y), col (x) of each mask pixel)
        mask_rows, mask_cols = np.where(mask)
        centroids.append(
            [
                (mask_cols.min() + mask_cols.max()) / 2,
                (mask_rows.min() + mask_rows.max()) / 2,
            ]
        )

    # new prompts as normalised (x, y)
    return np.array(centroids, dtype=np.float32).reshape(-1, 2) / [
        img_w,
        img_h,
    ]


def convert_bool_to_id_mask(list_masks, img_h, img_w):
    """Express a list of boolean masks as an ID-encoded mask.

    ``list_masks`` is a length-N list of (img_h, img_w) boolean masks; the
    returned ``id_mask`` has shape (img_h, img_w) and holds ID ``i + 1`` at
    the pixels of mask ``i`` (0 = background). Higher ID wins on overlap.
    """
    # initialise the ID-encoded mask as all background (0)
    id_mask = np.zeros((img_h, img_w), dtype=np.int16)

    # write each mask's ID into its pixels (higher ID wins on overlap)
    region_ids = np.arange(1, len(list_masks) + 1, dtype=np.int16)
    for region_id, bool_mask in zip(region_ids, list_masks, strict=True):
        id_mask[bool_mask] = region_id

    return id_mask, region_ids


def convert_id_mask_to_bool(id_mask):
    """Express an ID-encoded mask as a boolean masks array.

    ``id_mask`` has shape (img_h, img_w) and holds one nonzero ID per region
    (0 = background); the returned ``bool_masks`` is an array of shape
    (N_masks, img_h, img_w), one boolean mask per nonzero ID.
    """
    # get the nonzero region IDs (0 = background)
    region_ids = np.unique(id_mask)
    region_ids = region_ids[region_ids != 0]

    # one boolean mask per region ID (empty (0, H, W) if no regions)
    bool_masks = id_mask[None] == region_ids[:, None, None]

    return bool_masks, region_ids


# %%
# Tiled inference
def _make_tiles(img_h, img_w, tile_side, tile_overlap) -> list[tuple[int]]:
    """Overlapping (x0, y0, x1, y1) tiles covering the image.

    standard image coordinates (origin top-left, y increasing downward).
    """
    step = tile_side - tile_overlap

    # we use set to easily remove tuple duplicates
    # (near the edge, the clamping behaviour means
    # many tiles can be "rounded" to the same tile)
    tiles = set()

    # loop thru candidate top left corner of tile
    for y0 in range(0, max(1, img_h - tile_overlap), step):
        for x0 in range(0, max(1, img_w - tile_overlap), step):
            # compute bottom right corner of tile,
            # clamped to image edge
            x1 = min(x0 + tile_side, img_w)
            y1 = min(y0 + tile_side, img_h)

            # add (x0,y0,x1,y1) to set
            tiles.add(
                (
                    # top left corner of tile; we
                    # derive it from bottom right corner
                    # so that tile always stays full sized
                    # (for interior tiles, it matches x0,y0)
                    max(0, x1 - tile_side),
                    max(0, y1 - tile_side),
                    # bottom right corner of tile
                    x1,
                    y1,
                )
            )
    return sorted(tiles)


def _run_sam3_points(image_pil, points_xy_norm, processor):
    """Run SAM3 (text + points) on a (cropped) image; masks in its frame."""
    # pass image
    inference_state = processor.set_image(image_pil)
    processor.reset_all_prompts(inference_state)

    # add text prompt
    if TEXT_PROMPT is not None:
        inference_state = processor.set_text_prompt(
            state=inference_state, prompt=TEXT_PROMPT
        )

    # add point prompts
    inference_state = add_point_prompts(
        processor,
        inference_state,
        points_xy_norm,
        labels=True,
    )

    # Get results
    masks = inference_state["masks"].cpu().numpy()
    scores = inference_state["scores"].float().cpu().numpy()
    del inference_state
    torch.cuda.empty_cache()

    masks = masks.squeeze(1) if masks.ndim == 4 else masks
    if masks.shape[0] == 0:
        return [], np.empty(0, dtype=float)

    # postprocess masks
    list_kept_masks, list_kept_scores, _drop_counts = postprocess_masks(
        masks,
        scores,
        MIN_MASK_AREA_PIXELS,
        MAX_MASK_AREA_PIXELS,
        MIN_SOLIDITY,
    )
    return list_kept_masks, list_kept_scores


def _is_duplicate(mask, kept_masks, iou_thresh, containment_thresh=0.8):
    """Return True if ``mask`` duplicates any kept mask by IoU/containment.

    A duplicate is flagged when, against any mask in ``kept_masks``, the IoU
    exceeds ``iou_thresh`` or the containment (fraction of the SMALLER of the
    two masks covered by the other) exceeds ``containment_thresh``.
    """
    mask_area = mask.sum()
    for k_mask in kept_masks:
        # IoU wrt existing mask
        inter_area = np.logical_and(mask, k_mask).sum()
        union_area = mask_area + k_mask.sum() - inter_area
        iou = inter_area / union_area if union_area else 0.0

        # containment wrt existing mask
        smaller_area = min(mask_area, k_mask.sum())
        containment = inter_area / smaller_area if smaller_area else 0.0

        if iou > iou_thresh or containment > containment_thresh:
            return True
    return False


def _merge_by_iou(masks, scores, iou_thresh, containment_thresh=0.8):
    """Greedy IoU de-duplication of full-image masks; higher score wins."""
    kept_masks, kept_scores = [], []
    for score_idx in np.argsort(scores)[::-1]:  # high score first
        single_mask = masks[score_idx]

        # if no duplicate of an already-kept mask: keep it
        if not _is_duplicate(
            single_mask, kept_masks, iou_thresh, containment_thresh
        ):
            kept_masks.append(single_mask)
            kept_scores.append(scores[score_idx])
    return kept_masks, np.array(kept_scores)


# %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
# Load frames as a lazy array and map each frame to its video

# Get image array
list_image_files = sorted(list(Path(images_dir).glob("*.png")))
image_array = ImageArrayLazy(list_image_files)
print(image_array.shape)

# Get video per image
list_video_per_img = [
    img_p.stem.split("_", 1)[0].split("-Loop")[0] for img_p in list_image_files
]


# %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
# Open the pass-1 ID-encoded masks zarr (read-only)
# Produced by notebook_burrows_sam3_pass_1.py; holds "masks" and "scores".

root_pass_1 = zarr.open_group(str(PASS_1_MASKS_ZARR), mode="a")
masks_pass_1 = root_pass_1["masks"]  # (n_images, H, W), int16 ID-encoded
scores_pass_1 = root_pass_1["scores"]  # (n_images, max_regions + 1), float32


# %%%%%%%%%%%%%%%%%%%%%%%%%%%%
# Initialise zarr for results of pass 2
n_images, image_h, image_w = image_array.shape[:3]

# Timestamp + base metadata reused when persisting the tiled (pass-2) store
root_pass_2, _ = initialise_mask_zarr(
    OUTPUT_DIR,
    images_dir,
    manual_prompts_csv,
    image_array.shape[1:3],
    TEXT_PROMPT,
    CONF_THRESHOLD,
    MIN_MASK_AREA_PIXELS,
    MAX_MASK_AREA_PIXELS,
    MIN_SOLIDITY,
    MAX_REGIONS_PER_IMAGE,
    data_pass=2,
)

# # add data origin (pass 1 or 2)
# root_pass_2.create_array(
#     "data_pass",
#     shape=scores_pass_1.shape,
#     chunks=(1, scores_pass_1.shape[1]),
#     dtype="int8",
#     fill_value=0,  # 0 = background / unused, 1 = pass-1, 2 = pass-2
# )


# %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
# Build SAM3 image model and processor

# Disable autograd for the whole notebook
torch.inference_mode().__enter__()

# to avoid mismatch between bfloat16 (from model params)
# and float (from input)
if torch.cuda.is_available():
    torch.autocast("cuda", dtype=torch.bfloat16).__enter__()

# Instiantiate model: holds weights, submodules
# (backbone, text encoder, grounding head, mask decoder)
# and forward pass operations
model = build_sam3_image_model()

# Instantiate the processor
# (a wrapper around the model, handling I/O conversion, building
# and mutation of the inference_state, prompt accumulation and
# confidence thresholding)
# NOTE: the model is accessible via processor.model
processor = Sam3Processor(model, confidence_threshold=CONF_THRESHOLD)


# %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
# Expand point prompts per video using predicted masks

# original point prompts in **normalised** coords, keyed by video string
points_xy_normalised_per_video = extract_normalised_point_prompts_per_video(
    manual_prompts_csv, image_w, image_h
)

# For each frame: derive extra prompts from the pass-1 mask centroids and
# append them to that video's manual prompts. Keyed by video string.
extended_prompts_xy_norm_per_video = {}
for frame_idx, video_str in enumerate(list_video_per_img):
    # if no manual prompts: continue
    prompts_img_norm = points_xy_normalised_per_video.get(video_str)
    if prompts_img_norm is None:
        continue

    # Get pass-1 masks as boolean masks, and scores for this frame
    # (scores are indexed by region ID with NaN where absent; dropping NaN
    # leaves them in ascending-ID order, aligned with the boolean masks)
    bool_masks, _ = convert_id_mask_to_bool(masks_pass_1[frame_idx])
    scores = scores_pass_1[frame_idx]
    scores = scores[~np.isnan(scores)]

    # Compute new prompt points from masks data
    # (result can be shape (0,2))
    new_points_xy_norm = compute_new_prompts_from_masks(
        bool_masks,
        scores,
        prompts_img_norm,
        PROMPT_SELECTION_MIN_SCORE,
        img_h=image_h,
        img_w=image_w,
    )
    extended_prompts_xy_norm_per_video[video_str] = np.vstack(
        [prompts_img_norm, new_points_xy_norm]
    )


# %%%%%%%%%%%%%%%%%
# Tiled inference seeded by old + new point prompts
# -----------------------------------------------------------
# The image is split into overlapping tiles; each tile runs SAM3 with
# the text prompt + whatever prompt points land inside it (transformed
# to tile coords). Tiles with no point exemplars are skipped.

# tile side in pixels
TILE_SIZE = int(image_h / 3)

# overlap between neighbouring tiles, in pixels
# we should aim for burrows diameter < overlap, so that at least each
# burrow is covered by one full tile
TILE_OVERLAP = int(TILE_SIZE / 2)  # int(image_w*0.05) #256

# IoU above which two tile masks are merged
MERGE_IOU = 0.5


# %%
# Run tiled inference
n_images, image_h, image_w = image_array.shape[:3]

for frame_idx, video_str in enumerate(list_video_per_img):
    # Get image
    img_full = image_array[frame_idx]

    # Get point prompts in image pixel coordinates (not normalised)
    prompts_img_norm = extended_prompts_xy_norm_per_video[video_str]
    prompts_img_px = prompts_img_norm * np.array(
        [image_w, image_h], dtype=np.float32
    )

    # Compute tiles
    tiles = _make_tiles(image_h, image_w, TILE_SIZE, TILE_OVERLAP)

    # Loop thru tiles
    list_tile_masks, list_tile_scores = [], []
    n_empty_tiles = 0
    for x0, y0, x1, y1 in tiles:
        # Check points within tile (comparing pixel coords)
        in_tile = (
            (prompts_img_px[:, 0] >= x0)
            & (prompts_img_px[:, 1] >= y0)
            & (prompts_img_px[:, 0] < x1)
            & (prompts_img_px[:, 1] < y1)
        )

        # If none, continue
        if not in_tile.any():
            n_empty_tiles += 1
            continue

        # Express prompt points in crop-local pixels,
        # then normalise to the crop size for SAM3
        prompts_crop_px = prompts_img_px[in_tile] - np.array(
            [x0, y0], dtype=np.float32
        )
        prompts_crop_norm = prompts_crop_px / np.array(
            [x1 - x0, y1 - y0], dtype=np.float32
        )

        # Run SAM3 on crop with normalised point prompts
        img_crop = Image.fromarray(img_full[y0:y1, x0:x1])
        list_bool_masks, list_scores = _run_sam3_points(
            img_crop,
            prompts_crop_norm,
            processor,
        )

        # Map each tile mask back into the full-image canvas
        for crop_mask in list_bool_masks:
            canvas_array = np.zeros((image_h, image_w), dtype=bool)
            # assign data from crop
            canvas_array[y0:y1, x0:x1] = crop_mask
            # add result to list of masks across all tiles
            list_tile_masks.append(canvas_array)

        # add to list of scores
        list_tile_scores.extend(list_scores.tolist())

    # Merge masks across tiles in this image
    merged_masks, merged_scores = _merge_by_iou(
        list_tile_masks,
        list_tile_scores,  # np.array(list_tile_scores, dtype=float),
        MERGE_IOU,
    )

    # Keep only MAX_REGIONS_PER_IMAGE (zarr shape limit)
    if len(merged_masks) > MAX_REGIONS_PER_IMAGE:
        print(
            f"Frame {frame_idx}: {len(list_tile_masks)} regions exceeds cap "
            f"({MAX_REGIONS_PER_IMAGE}), clipping"
        )
        merged_masks = merged_masks[:MAX_REGIONS_PER_IMAGE]
        merged_scores = merged_scores[:MAX_REGIONS_PER_IMAGE]

    # Compute id-encoded mask per region
    # boolean masks (N, H, W) -> ID-encoded (H, W); higher ID wins on overlap
    id_mask, region_ids = convert_bool_to_id_mask(
        merged_masks, image_h, image_w
    )

    # Save ID-encoded masks for this image
    root_pass_2["masks"][frame_idx] = id_mask
    root_pass_2["scores"][frame_idx, region_ids] = merged_scores

    # Delete big per image tile lists?
    # ...


# %%
# Combine pass-1 and pass-2 into a fresh store (cross-pass IoU dedup)

# Pass-1 masks are kept unconditionally; a pass-2 mask is dropped if it
# duplicates an already-kept mask (pass 1, or an earlier-kept pass 2) by IoU
# or containment. Provenance is recorded in a "data_pass" array (1 = pass-1,
# 2 = pass-2), indexed by region ID exactly like "scores".

# IoU above which a pass-2 mask is treated as a duplicate of a kept mask
CROSS_PASS_IOU = MERGE_IOU

# MAX_REGIONS_PER_IMAGE for the combined store
COMBINED_MAX_REGIONS = 1.5 * MAX_REGIONS_PER_IMAGE # ok?

# Fresh combined store (same schema as the per-pass stores)
root_combined, output_combined_zarr = initialise_mask_zarr(
    OUTPUT_DIR,
    images_dir,
    manual_prompts_csv,
    image_array.shape[1:3],
    TEXT_PROMPT,
    CONF_THRESHOLD,
    MIN_MASK_AREA_PIXELS,
    MAX_MASK_AREA_PIXELS,
    MIN_SOLIDITY,
    COMBINED_MAX_REGIONS,
    data_pass="combined",
)

# Provenance array
# 0 = unused, 1 = pass-1, 2 = pass-2
data_pass_arr = root_combined.create_array(
    "data_pass",
    shape=root_combined["scores"].shape,
    chunks=(1, root_combined["scores"].shape[1]),
    dtype="int8",
    fill_value=0,
)

# %%
# Get results of  pass 2
masks_pass_2 = root_pass_2["masks"]
scores_pass_2 = root_pass_2["scores"]

# Loop thru images
for frame_idx in range(n_images):
    # Pass-1 masks are kept unconditionally
    # (scores indexed by region ID, aligned with the ascending-ID bool masks)
    bool_1, ids_1 = convert_id_mask_to_bool(masks_pass_1[frame_idx])
    kept_masks = list(bool_1)
    kept_scores = list(scores_pass_1[frame_idx, ids_1])
    kept_pass = [1] * len(kept_masks)

    # Pass-2 masks added only if not a duplicate of an already-kept mask;
    # highest score first so the strongest of any near-duplicates survives
    bool_2, ids_2 = convert_id_mask_to_bool(masks_pass_2[frame_idx])
    scr_2 = scores_pass_2[frame_idx, ids_2]
    for i in np.argsort(scr_2)[::-1]:
        if not _is_duplicate(bool_2[i], kept_masks, CROSS_PASS_IOU):
            kept_masks.append(bool_2[i])
            kept_scores.append(scr_2[i])
            kept_pass.append(2)

    # Encode and write masks + scores + provenance, all indexed by region ID
    # (IDs follow kept_masks order: pass-1 IDs first, then kept pass-2)
    id_mask, region_ids = convert_bool_to_id_mask(
        kept_masks, image_h, image_w
    )
    root_combined["masks"][frame_idx] = id_mask
    root_combined["scores"][frame_idx, region_ids] = kept_scores
    data_pass_arr[frame_idx, region_ids] = kept_pass

print(f"Saved combined ID-mask zarr to {output_combined_zarr}")




# %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
# --- plot: pass-1 vs tiled result -----------------------------------------
# Read both ID-masks back from zarr so we don't hold N full-res bool masks
# in RAM, and use imshow instead of N ax.contour calls (much lighter).
pass1_id_mask = masks_pass_1[selected_frame_idx]
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
pass1_id_mask = masks_pass_1[selected_frame_idx]
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
