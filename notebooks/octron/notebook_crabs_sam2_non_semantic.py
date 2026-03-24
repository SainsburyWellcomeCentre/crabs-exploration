"""RUn SAM2 to generate crab masks.

Standalone script: pass bounding boxes as prompts to SAM2 via OCTRON, predict
masks only in those regions, then export as YOLO detection training data.

Input: a directory of PNG frames (sorted lexicographically as the frame sequence).

BBox format: (N, 4) float32 array of [x1, y1, x2, y2] in pixel coordinates
  x = column (width axis), y = row (height axis)

Dependencies are auto-installed if the script is executed via uv run
"""

# /// script
# requires-python = ">=3.11"
# dependencies = [
#   "Pillow",
#   "zarr",
#   "ethology",
#   "sam2 @ git+https://github.com/facebookresearch/sam2.git",
# ]
# ///

# %%
from datetime import datetime
from pathlib import Path

import numpy as np
import torch
import zarr
from ethology.io.annotations import load_bboxes
from PIL import Image
from sam2.sam2_image_predictor import SAM2ImagePredictor

# %%%%%%%%%%%%%%%
# Input data

# September groundtruth data
DATA_DIR = Path("/Users/sofia/swc/CrabLabels/sep2023-full")
IMAGES_DIR = DATA_DIR / "frames"
ANNOTATIONS_DIR = DATA_DIR / "annotations"
ANNOTATIONS_FILE = ANNOTATIONS_DIR / "VIA_JSON_combined_coco_gen.json"


IMG_BATCH_SIZE = 4
MODEL_ID = "facebook/sam2.1-hiera-base-plus"

# %%%%%%%%%%%%%%%%%%
# Output

# Zarr output dir
timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

output_masks_zarr = ANNOTATIONS_DIR / f"masks_{timestamp}.zarr"
output_masks_zarr.mkdir(parents=True, exist_ok=True)


# %%%%%%%%%%%%%%%%%%%%%%%%%%
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


# %%%%%%%%%%%%%%%%%
# Set device
if torch.cuda.is_available():
    device = "cuda"
elif torch.backends.mps.is_available():
    device = "mps"
else:
    device = "cpu"
print(f"Using device: {device}")


# %%%%%%%%%%%%%%%%%%%%%%
# Read groundtruth as ethology annotation dataset
ds_bboxes = load_bboxes.from_files(
    ANNOTATIONS_FILE,
    format="COCO",
    images_dirs=IMAGES_DIR,
)

# %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
# Load ground truth images as lazy array
# build lazy image array
png_files = sorted(ds_bboxes.attrs["images_directories"].glob("*.jpg"))
image_array = ImageArrayLazy(png_files)

# add as an attribute
ds_bboxes.attrs["image_array"] = image_array


# %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
# Load SAM2 predictor on device
image_predictor = SAM2ImagePredictor.from_pretrained(
    model_id=MODEL_ID,
    device=device,
)


# %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
# Define per-frame exemplar bboxes from groundtruth

# corner 1 is min x, min y
# corner 2 is max x, max y
x1y1 = ds_bboxes.position - ds_bboxes.shape / 2
x2y2 = ds_bboxes.position + ds_bboxes.shape / 2

# Each key is a frame index;
# value is an (N, 4) float32 array [x1, y1, x2, y2].
map_frame_idx_to_boxes = {
    idx: np.c_[
        x1y1.sel(image_id=idx).dropna(dim="id", how="all").values.T,
        x2y2.sel(image_id=idx).dropna(dim="id", how="all").values.T,
    ]
    for idx in range(len(ds_bboxes.image_id))
}

# %%%%%%%%%%%%%%%%%%%%%%%%%%%
# Initialise zarr store for output masks

# Create zarr store for integer masks (OCTRON-like)
n_images = ds_bboxes.attrs["image_array"].shape[0]
image_h = ds_bboxes.attrs["image_array"].shape[1]
image_w = ds_bboxes.attrs["image_array"].shape[2]

mask_zarr = zarr.open(
    output_masks_zarr,
    mode="w",
    shape=(n_images, image_h, image_w),
    dtype="bool",
    fill_value=False,
    chunks=(1, image_h, image_w),
)

# %%%%%%%%%%%%%%%%%%%%%%%%%%
# Add metadata to zarr store
mask_zarr.attrs.update(
    {
        "timestamp": timestamp,
        "sam2_model": MODEL_ID,
        "source_images_dir": str(IMAGES_DIR),
        "annotations_file": str(ANNOTATIONS_FILE),
        "annotations_format": "COCO",
        "n_images": n_images,
        "image_shape": [image_h, image_w],
        "batch_size": IMG_BATCH_SIZE,
        "multimask_output": False,
        "prompt_type": "bounding_box",
        "mask_encoding": "instance_id",  # vs "binary"
        "background_label": 0,
        "id_offset": 1,  # mask_label = bbox_id + 1
        "device": device,
    }
)


# %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
# Process images in batches, predict masks, save to zarr store

batch_size = IMG_BATCH_SIZE
n_frames = len(ds_bboxes.image_id)
img_h, img_w = ds_bboxes.attrs["image_array"].shape[1:3]

# loop thru images
for idx in range(0, n_frames, batch_size):
    # Adjust batch size if required
    actual_batch_size = min(batch_size, n_frames - idx)

    # Initialise id_mask for this batch
    # an integer array (B, H, W) where each pixel stores which object "owns" it
    # 0 = background, rest are 1-based object instance IDs
    id_mask_batch = np.zeros((actual_batch_size, img_h, img_w), dtype=np.int16)

    # Compute embeddings for image batch
    image_batch = [image_array[idx + i] for i in range(actual_batch_size)]
    image_predictor.set_image_batch(image_batch)

    # Compute list of of bboxes — list of (N_i, 4) arrays, one per image
    boxes_batch = [
        map_frame_idx_to_boxes[f_i]
        for f_i in range(idx, idx + actual_batch_size)
    ]

    # Predict batch of masks
    # masks_batch is a list — one (N, 1, H, W) array per image
    # where N is number of boxes
    masks_batch, scores_batch, _ = image_predictor.predict_batch(
        box_batch=boxes_batch,
        multimask_output=False,
    )

    # Convert boolean masks to ID-encoded masks OCTRON expects
    # (higher ID wins in overlap)
    for idx_rel_batch in range(actual_batch_size):
        # Get masks for one frame
        masks_one_frame = masks_batch[idx_rel_batch].squeeze(
            axis=1
        )  # (N, H, W)

        # Convert boolean mask to 1-based integer mask per object ID
        n_objects = masks_one_frame.shape[0]
        obj_ids = np.arange(1, n_objects + 1, dtype=np.int16)[:, None, None]
        id_mask_batch[idx_rel_batch] = (masks_one_frame * obj_ids).max(axis=0)

        print(
            f"Frame {idx + idx_rel_batch}: "
            f"{n_objects} masks / {boxes_batch[idx_rel_batch].shape[0]} boxes"
        )

    # Save id_mask to zarr store
    mask_zarr[idx : idx + actual_batch_size] = id_mask_batch

    # Mark frames as annotated in zarr store attributes
    annotated = set(mask_zarr.attrs.get("annotated_frames", []))
    annotated.update(range(idx, idx + actual_batch_size))
    mask_zarr.attrs["annotated_frames"] = sorted(annotated)


print(f"Saved ID-encoded mask zarr to {output_masks_zarr}")
