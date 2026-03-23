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
#   "sparse",
# ]
# ///
# %%
from datetime import datetime
from pathlib import Path

import napari
import numpy as np
import zarr
from ethology.io.annotations import load_bboxes
from octron.sam_octron.helpers.sam2_zarr import (
    create_image_zarr,
    mark_frames_annotated,
)
from PIL import Image
from sam2.sam2_image_predictor import SAM2ImagePredictor
from skimage.measure import regionprops
from ultralytics import YOLO

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
    """Return ellipse corners, major-axis lines, and minor-axis lines.

    Each ellipse is represented by the 4 corners of its oriented bounding
    rectangle (for napari's built-in ``"ellipse"`` shape type).

    Returns
    -------
    ellipse_corners : list of (4, 2) arrays
        Corners in (y, x) order.
    major_axes : list of (2, 2) arrays
        Endpoint pairs for the major axis (y, x).
    minor_axes : list of (2, 2) arrays
        Endpoint pairs for the minor axis (y, x).
    """
    ellipse_corners = []
    major_axes = []
    minor_axes = []
    for prop in regionprops(label_image):
        cy, cx = prop.centroid
        a = prop.major_axis_length / 2
        b = prop.minor_axis_length / 2
        theta = prop.orientation
        # scikit-image orientation: angle of the major axis
        # measured counter-clockwise from the row (y) axis
        # direction vectors in (y, x)
        dir_major = np.array([np.cos(theta), np.sin(theta)])
        dir_minor = np.array([-np.sin(theta), np.cos(theta)])

        center = np.array([cy, cx])

        # 4 corners of the oriented bounding rectangle
        corners = np.array(
            [
                center + a * dir_major + b * dir_minor,
                center + a * dir_major - b * dir_minor,
                center - a * dir_major - b * dir_minor,
                center - a * dir_major + b * dir_minor,
            ]
        )
        ellipse_corners.append(corners)

        # axis endpoints
        major_axes.append(
            np.array([center - a * dir_major, center + a * dir_major])
        )
        minor_axes.append(
            np.array([center - b * dir_minor, center + b * dir_minor])
        )

    return ellipse_corners, major_axes, minor_axes


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
# Load SAM2 predictor
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
import dask.array as da
import sparse
import xarray as xr

output_masks_dir = Path(
    "/Users/sofia/arc/project_Zoo_crabs/crabs-exploration/output/sep2023-full"
)

zarr_root = zarr.open(
    output_masks_dir / f"{LABEL_NAME} masks.zarr",
    mode="r",
)
# mask_array = zarr_root["masks"]

mask_da_array = da.from_zarr(zarr_root["masks"])


# %%%%%%%%%%%%%%%%
# Can I add masks to ds_bboxes? aligned with ids?


# Option 1: as mask arrays
# (but we lose alignment with ID) --- except if ID matches label?
ds_bboxes.assign_coords(
    {
        "img_h": np.arange(mask_da_array.shape[1]),
        "img_w": np.arange(mask_da_array.shape[2]),
    }
)

ds_bboxes["masks"] = xr.DataArray(
    data=mask_da_array, dims=["image_id", "img_h", "img_w"]
)


# %%
# Option 2a: store sparse array objects?
# Is this faster than option b?

# initialise array of sparse masks
sparse_mask_array = np.empty(
    (ds_bboxes.sizes["image_id"], ds_bboxes.sizes["id"]),
    dtype=object,
)

# Fill in values
# ATT! bboxes id is 0-based
# masks's id is 1-based!
# we assume they are in the same order?
for bbox_id in np.arange(ds_bboxes.sizes["id"]):
    # compute 3D coords for this ID
    coords = da.nonzero(mask_da_array == bbox_id + 1)  # id_0, id_1, id_2

    # express as numpy
    coords = [c.compute() for c in coords]

    # compute sparse array for this ID
    s = sparse.COO(
        coords, bbox_id, shape=mask_da_array.shape
    )  # shape 544, 2160, 4096

    # split sparse array per image_id
    s_per_image_id = sparse.unstack(s, axis=0)  # tuple of 544 slices

    # assign to array
    sparse_mask_array[:, bbox_id] = s_per_image_id


ds_bboxes["masks_sparse"] = xr.DataArray(
    data=sparse_mask_array,
    dims=["image_id", "id"],
)

# %%
# Option 2b: store sparse array objects?
# Q: should I used masked arrays rather than sparse arrays?

sparse_mask_array = np.empty(
    (ds_bboxes.sizes["image_id"], ds_bboxes.sizes["id"]),
    dtype=object,
)

for image_id in range(mask_da_array.shape[0]):
    # Get rows, cols and values of non-zero elements
    # in this frame
    mask_frame = mask_da_array[image_id].compute()  # load one frame at a time
    rows, cols = np.nonzero(mask_frame)
    labels = mask_frame[rows, cols]

    # define sparse array for this frame
    for l_i, lbl in enumerate(np.unique(labels)):
        slc_nnz = labels == lbl
        s = sparse.COO(
            np.c_[rows[slc_nnz], cols[slc_nnz]].T, lbl, shape=mask_frame.shape
        )  # img_h, img_w

        # assign to full sparse array
        sparse_mask_array[image_id, l_i] = s

ds_bboxes["masks_sparse"] = xr.DataArray(
    data=sparse_mask_array,
    dims=["image_id", "id"],
)

# %%
# plot image with a bbox and mask
import matplotlib.pyplot as plt

image_id = 40
id = 10

fig, ax = plt.subplots(1, 1)
# image
ax.imshow(ds_bboxes.image_array[image_id])
# bbox centre
ax.scatter(
    ds_bboxes.position.sel(image_id=image_id, id=id, space="x"),
    ds_bboxes.position.sel(image_id=image_id, id=id, space="y"),
    15,
    marker="x",
    color='r'
)
# single mask
# Q: should I used masked arrays rather than sparse arrays?
mask_dense = ds_bboxes.masks_sparse.isel(image_id=image_id, id=id).item().todense()
ax.imshow(
    np.ma.masked_where(mask_dense == 0, mask_dense), cmap="turbo", alpha=0.5
)

# %%
# plot all masks in one frame
image_id = 40

fig, ax = plt.subplots(1, 1)
# image
ax.imshow(ds_bboxes.image_array[image_id])

all_masks_per_nonempty_id  = sparse.stack(
    ds_bboxes.masks_sparse.isel(image_id=image_id).dropna(dim='id').values
) # (non-empty id, img_h, img_w)
all_masks_dense = all_masks_per_nonempty_id.max(axis=0).todense()
ax.imshow(
    np.ma.masked_where(all_masks_dense == 0, all_masks_dense), cmap="turbo", alpha=0.5,
)

# %%
# %matplotlib widget
# %%
# What I would like is
# ds_bboxes.masks.sel(image_id=0) ---> masks for that frame
# ds_bboxes.masks.sel(image_id=0, id=3) ---> mask for that frame and id=3


# %%
# Option 3: store dask array objects?
mask_da = np.empty(ds_bboxes.sizes["image_id"])


ds_bboxes["masks_da"] = xr.DataArray(
    data=mask_da,
    dims=["image_id", "id"],
)


# %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
# Load frames and masks in napari viewer
viewer = napari.Viewer()
# viewer.add_image(np.asarray(image_array).moveaxis(0, -1, 1, 2), name="image")
viewer.add_labels(np.asarray(mask_array), name=f"{LABEL_NAME} masks")

# %%
# build ellipses and axis lines for napari Shapes layers
all_ellipses = []
all_major_axes = []
all_minor_axes = []
for frame_idx in range(mask_array.shape[0]):
    ellipse_corners, major_axes, minor_axes = ellipses_from_labels(
        np.asarray(mask_array[frame_idx])
    )
    for corners in ellipse_corners:
        # Prepend frame index as first column for nD shapes
        nd = np.column_stack(
            [np.full(corners.shape[0], fill_value=frame_idx), corners]
        )
        all_ellipses.append(nd)
    for axis in major_axes:
        nd = np.column_stack(
            [np.full(axis.shape[0], fill_value=frame_idx), axis]
        )
        all_major_axes.append(nd)
    for axis in minor_axes:
        nd = np.column_stack(
            [np.full(axis.shape[0], fill_value=frame_idx), axis]
        )
        all_minor_axes.append(nd)

viewer.add_shapes(
    all_ellipses,
    shape_type="ellipse",
    edge_color="yellow",
    edge_width=4,
    face_color="transparent",
    name="ellipses",
)
viewer.add_shapes(
    all_minor_axes,
    shape_type="line",
    edge_color="green",
    edge_width=4,
    name="minor axes",
)
viewer.add_shapes(
    all_major_axes,
    shape_type="line",
    edge_color="red",
    edge_width=4,
    name="major axes",
)


# %%%%%%%%%%%%%%%%%%%%%%%
# Generate YOLO detection training data via OCTRON

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
            "frames": mask_array.attrs["annotated_frames"],
            "masks": [mask_array],
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
    training_fraction=0.8,
    validation_fraction=0.2,
    verbose=True,
)

# %%
# Write images and YOLO label file
for _ in yolo.create_training_data_segment():
    pass

# # Replace copied PNGs with symlinks to originals
# for split_dir in yolo.data_path.iterdir():
#     for img_file in split_dir.glob("*.png"):
#         # Would the frame_id actually work?
#         frame_id = int(img_file.stem.split("_")[-1])
#         original = png_files[frame_id]
#         img_file.unlink()
#         img_file.symlink_to(original)

# %%
# Write config
yolo.write_yolo_config(train_mode="segment")

# - yolo.data_path has the YOLO segmentation dataset
# - yolo.config_path has the data.yaml
# %%
# Training

segmentor_model = YOLO("yolo11n-seg.pt")
segmentor_model.train(
    data=str(yolo.config_path),  # path to data.yaml
    epochs=100,
    imgsz=1280,  # 1600? 2144?
)
# %%
# Inference via OCTRON, or Sahi?


# Sahi:
# from sahi import AutoDetectionModel
# from sahi.predict import get_sliced_prediction

# model = AutoDetectionModel.from_pretrained(
#     model_type="ultralytics",
#     model_path="path/to/best.pt",
#     confidence_threshold=0.5,
#     device="mps",
# )

# result = get_sliced_prediction(
#     image="path/to/frame.png",
#     detection_model=model,
#     slice_height=1024,
#     slice_width=1024,
#     overlap_height_ratio=0.2,
#     overlap_width_ratio=0.2,
# )
