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
    normalize_bbox,  # TODO: replace with equivalent numpy function
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
TEXT_PROMPT = (
    "hole"  # "crab burrow in sand"  # set to None to skip the text prompt
)
CONF_THRESHOLD = (
    0.35  # masks below this threshold are not scaled up to full res
)

# -------------------------
# Postprocessing of masks
# -------------------------
MIN_MASK_AREA_PIXELS = 200  # 100  # 200? in crab bodylengths?
MAX_MASK_AREA_PIXELS = 2500
MIN_SOLIDITY = 0.95

# Output dir for masks
# TODO: add timestamp
OUTPUT_DIR = Path(
    "/home/sminano/swc/project_crabs/crabs-exploration/output_burrows_sam3"
)

MAX_REGIONS_PER_IMAGE = 500

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
):
    """Create mask ID encoded zarr store timestamped and with metadata."""
    # Create a timestamped masks zarr store in the output directory
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir.mkdir(parents=True, exist_ok=True)
    output_masks_zarr = output_dir / f"masks_{timestamp}.zarr"

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


def add_prompts_to_inference_state(
    processor, image, prompts_xy_norm, text_prompt=None
):
    """Add iamge, normalised points and text prompts to inference state."""
    # Pass image to processor and reset inference state
    inference_state = processor.set_image(image)
    processor.reset_all_prompts(
        inference_state
    )  # mutates the state dict in place

    # Add optional text prompt
    if text_prompt is not None:
        inference_state = processor.set_text_prompt(
            state=inference_state, prompt=text_prompt
        )

    # Add normalised point prompts (all at once; grounds a single time)
    inference_state = add_point_prompts(
        processor, inference_state, prompts_xy_norm, labels=True
    )

    return inference_state


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


def extract_sam3_results_one_img(inference_state):
    """Extract SAM3 boolean masks and scores for a single image.

    Move masks to CPU, then release this frame's GPU state before the
    next frame: otherwise the previous state (backbone features +
    full-res masks_logits) stays alive during the next forward pass.
    """
    masks = inference_state["masks"].cpu().numpy()
    scores = inference_state["scores"].float().cpu().numpy()  # (N,)

    masks = masks.squeeze(1) if masks.ndim == 4 else masks  # (N, H, W)
    n_objects = masks.shape[0]

    return masks, scores, n_objects


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


def convert_bool_to_id_mask(list_kept_region_masks, img_h, img_w):
    """Express boolean masks array as ID encoded mask.

    Higher ID wins on overlap.
    """
    # initialise id-encode mask with all zeros
    id_mask = np.zeros((img_h, img_w), dtype=np.int16)

    # loop thru region IDs
    region_ids = np.arange(1, len(list_kept_region_masks) + 1, dtype=np.int16)
    for region_id, bool_mask in zip(
        region_ids,
        list_kept_region_masks,
        strict=True,
    ):
        id_mask[bool_mask] = region_id

    return id_mask, region_ids


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
# Initialise the output ID-encoded mask zarr store
# (includes scores)

image_shape = image_array.shape[:3]
root, output_masks_zarr = initialise_mask_zarr(
    OUTPUT_DIR,
    images_dir,
    manual_prompts_csv,
    image_shape,
    TEXT_PROMPT,
    CONF_THRESHOLD,
    MIN_MASK_AREA_PIXELS,
    MAX_MASK_AREA_PIXELS,
    MIN_SOLIDITY,
    MAX_REGIONS_PER_IMAGE,  # ok?
)


# %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
# Extract point prompts per video
# CSV columns: group_id, prompt_point_x, prompt_point_y
# group_id has format "<video>_mean_n<frame>.png"; the video string is
# everything before "_mean".


# point prompts in **normalised** coords, keyed by video string
points_xy_normalised_per_video = extract_normalised_point_prompts_per_video(
    manual_prompts_csv, image_array.img_w, image_array.img_h
)

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


# %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
# Run inference on every frame and write ID-encoded masks to zarr

count_postproc_frames_empty = 0
postproc_frames_w_masks = []

for frame_idx in range(len(image_array)):
    # Load image
    image = Image.fromarray(image_array[frame_idx])
    img_w, img_h = image.size

    # Get corresponding video
    video_str = list_video_per_img[frame_idx]

    # Get point prompts normalised for that video
    prompts_xy_norm = points_xy_normalised_per_video.get(video_str)
    if prompts_xy_norm is None or len(prompts_xy_norm) == 0:
        print(f"Frame {frame_idx} ({video_str}): no prompts, skipping")
        continue

    # Add all prompts to inference state
    inference_state = add_prompts_to_inference_state(
        processor,
        image,
        prompts_xy_norm,
        text_prompt=TEXT_PROMPT,
    )

    # Get predicted boolean masks and scores
    masks, scores, n_objects = extract_sam3_results_one_img(inference_state)
    del inference_state
    torch.cuda.empty_cache()

    if n_objects == 0:
        print(f"Frame {frame_idx} ({video_str}): no detections")
        continue

    # Split masks into "regions" and postprocess
    list_kept_region_masks, list_kept_scores, drop_counts = postprocess_masks(
        masks,
        scores,
        MIN_MASK_AREA_PIXELS,
        MAX_MASK_AREA_PIXELS,
        MIN_SOLIDITY,
    )

    # -------------------------------------------
    # Log postprocessing results for this frame
    # (n_total_regions: total regions split from masks)
    if not list_kept_region_masks:
        print(
            f"Frame {frame_idx} ({video_str}): no masks after postprocessing"
        )
        count_postproc_frames_empty += 1
        continue

    n_kept_regions = len(list_kept_region_masks)
    n_total_regions = n_kept_regions + sum(drop_counts.values())
    print(
        f"Frame {frame_idx} ({video_str}): postprocessing kept "
        f"{n_kept_regions}/{n_total_regions} regions, "
        f"dropped {drop_counts}"
    )

    # ------------------------------------
    # Compute id-encoded mask per region
    # boolean masks (N, H, W) -> ID-encoded (H, W); higher ID wins on overlap
    id_mask, region_ids = convert_bool_to_id_mask(
        list_kept_region_masks, img_h, img_w
    )

    if region_ids.max() > MAX_REGIONS_PER_IMAGE:
        print(
            f"Frame {frame_idx}: {len(region_ids)} regions exceeds cap"
            # will fail on write
        )

    # Save results to zarr
    root["masks"][frame_idx] = id_mask
    root["scores"][frame_idx, region_ids] = list_kept_scores

    # Log frames with masks that survived
    postproc_frames_w_masks.append(frame_idx)

    print(
        f"Frame {frame_idx} ({video_str}): {len(list_kept_region_masks)} masks"
    )

print(f"Saved ID-encoded mask zarr to {output_masks_zarr}")

# Log frames with final masks
root.attrs["frames_with_masks"] = postproc_frames_w_masks

print(f"Frames with no masks: {count_postproc_frames_empty}")

