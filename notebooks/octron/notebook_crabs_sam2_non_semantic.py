"""SAM2 for crab masks via OCTRON.

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
#   "octron[all] @ git+https://github.com/OCTRON-tracking/OCTRON-GUI.git",
#   "ethology",
# ]
# ///
# %%
import math
from datetime import datetime
from pathlib import Path

import napari
import numpy as np
import xarray as xr
import zarr
from ethology.io.annotations import load_bboxes
from octron.sam_octron.helpers.sam2_zarr import (
    create_image_zarr,
    mark_frames_annotated,
)
from PIL import Image
from sam2.sam2_image_predictor import SAM2ImagePredictor
from skimage.measure import regionprops

# Input data


# Predictor
SAM_OCTRON_LOCAL_DIR = Path(
    "/Users/sofia/swc/project_octron/OCTRON-GUI/octron/sam_octron"
)
SAM2_CKPT_PATH = (
    SAM_OCTRON_LOCAL_DIR / "checkpoints" / "sam2.1_hiera_base_plus.pt"
)
SAM2_CONFIG_PATH = (
    SAM_OCTRON_LOCAL_DIR / "configs" / "sam2.1" / "sam2.1_hiera_b+.yaml"
)

# September groundtruth data
DATA_DIR = Path("/Users/sofia/swc/CrabLabels/sep2023-full")
IMAGES_DIR = DATA_DIR / "frames"
ANNOTATIONS_FILE = DATA_DIR / "annotations" / "VIA_JSON_combined_coco_gen.json"

# Label name
# (used for output naming only, SAM2 doesn't do text grounding)
LABEL_NAME = "crab"

# Output dir
timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
OUTPUT_DIR = Path(
    f"/Users/sofia/arc/project_Zoo_crabs/crabs-exploration/output_{timestamp}"
)  # root output folder

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


def ellipses_from_labels(label_image):
    """Return list of ellipse arrays for napari Shapes layer."""
    ellipses_yx = []
    for prop in regionprops(label_image):
        cy, cx = prop.centroid
        a = prop.major_axis_length / 2
        b = prop.minor_axis_length / 2
        theta = prop.orientation
        # in radians, counter-clockwise from horizontal
        # scikit-image's orientation is measured from the row axis
        # Sample points around the ellipse
        t = np.linspace(0, 2 * math.pi, 60)
        ey = cy + a * np.cos(t) * np.cos(theta) - b * np.sin(t) * np.sin(theta)
        ex = cx + a * np.cos(t) * np.sin(theta) + b * np.sin(t) * np.cos(theta)
        ellipses_yx.append(np.column_stack([ey, ex]))
    return ellipses_yx


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
png_files = sorted(ds_bboxes.attrs["images_directories"].glob("*.png"))
image_array = ImageArrayLazy(png_files)
print(image_array.shape)

# add as attribute?
ds_bboxes.attrs["image_array"] = image_array


# %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
# Load predictor
image_predictor = SAM2ImagePredictor.from_pretrained(
    "facebook/sam2.1-hiera-base-plus", device="mps"
)
# model, device = build_sam2_octron(
#     config_file_path=SAM2_CONFIG_PATH,
#     ckpt_path=SAM2_CKPT_PATH,
# )

# image_predictor = SAM2ImagePredictor(model)


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
data_id_str = DATA_DIR.name
output_masks_dir = OUTPUT_DIR / data_id_str
output_masks_dir.mkdir(parents=True, exist_ok=True)

mask_zarr = create_image_zarr(
    zarr_path=output_masks_dir / f"{LABEL_NAME} masks.zarr",
    num_frames=ds_bboxes.attrs["image_array"].shape[0],
    image_height=ds_bboxes.attrs["image_array"].shape[1],
    image_width=ds_bboxes.attrs["image_array"].shape[2],
    fill_value=-1,
    dtype="int16",
    video_hash_abbrev=data_id_str,
)


# %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
# Process images in batches, predict masks, save to zarr store

batch_size = 2  # samples
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
    for frame_idx in range(idx, idx + actual_batch_size):
        mark_frames_annotated(mask_zarr, frame_idx)

print(
    f"Saved ID-encoded mask zarr to {output_masks_dir / f'{LABEL_NAME} masks.zarr'}"
)

# %%%%%%%%%%%%%%%%%%%%%%%
# Load masks from zarr store

# mask_store = zarr.open_group(
#     annotation_dir / f"{LABEL_NAME} masks.zarr", mode="r"
# )
# mask_data = mask_store["masks"]

# mask_data = xr.open_zarr(
#     "/Users/sofia/arc/project_Zoo_crabs/crabs-exploration/output/sep2023-full/crab masks.zarr"
# )
zarr_groups = zarr.open(output_masks_dir / f"{LABEL_NAME} masks.zarr", mode="r")
mask_data = zarr_groups["masks"]

# %%%%%%%%%%%%%%%%%%%%%
# Load frames and masks in napari viewer
viewer = napari.Viewer()
viewer.add_image(image_array, name="image")
viewer.add_labels(np.asarray(mask_data), name=f"{LABEL_NAME} masks")


# build all ellipses array for napari Shapes layer
all_ellipses = []
for frame_idx in range(mask_data.shape[0]):
    frame_ellipses = ellipses_from_labels(np.asarray(mask_data[frame_idx]))
    for ell in frame_ellipses:
        # Prepend frame index as first column for nD shapes
        nd_ell = np.column_stack(
            [
                np.full(ell.shape[0], fill_value=frame_idx),
                ell,
            ]
        )
        all_ellipses.append(nd_ell)

viewer.add_shapes(
    all_ellipses,
    shape_type="polygon",
    edge_color="yellow",
    face_color="transparent",
    name="ellipses",
)

# %%%%%%%%%%%%%%%%
# Can I add masks to ds_bboxes? aligned with ids?

# %%%%%%%%%%%%%%%%%%%%%%%
# Generate YOLO detection training data.

from octron.yolo_octron.yolo_octron import YOLO_octron

yolo = YOLO_octron(project_path=OUTPUT_DIR, clean_training_dir=True)
yolo.train_mode = "segment"

# Add label dict
yolo.label_dict = {
    output_masks_dir.as_posix(): {
        # label description
        # key is class_ID
        0: {
            "label": LABEL_NAME,  # class name for YOLO
            "original_id": 1,  # from napari
            "frames": mask_data.attrs["annotated_frames"],
            "masks": [mask_data],
            "color": [1.0, 0.0, 0.0, 1.0],  # for GUI only
        },
        # session metadata
        "video": image_array,
        "video_file_path": IMAGES_DIR,
    }
}

# %%
# Extract polygons from masks
# Check the output: each label
# entry in label_dict should now have a "polygons" key.

# it's a generator so we need to consume it
# for it to run
yolo.enable_watershed = False
for _ in yolo.prepare_polygons():
    pass

# %%
# Train / val/ test split
yolo.prepare_split(
    training_fraction=0.7,
    validation_fraction=0.15,
    verbose=True,
)

# %%
# Write images and YOLO label file
for _ in yolo.create_training_data_segment():
    pass

# %%
# Write config
yolo.write_yolo_config(train_mode="segment")

# yolo.data_path has the YOLO segmentation dataset
# yolo.config_path has the data.yaml
# %%
