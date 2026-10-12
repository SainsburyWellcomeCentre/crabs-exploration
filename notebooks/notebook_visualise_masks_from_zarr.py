"""A notebook to visualise masks in annotations dataset."""

# %%
from pathlib import Path

import dask
import dask.array as da
import matplotlib.pyplot as plt
import napari
import numpy as np
import xarray as xr
import zarr
from ethology.io.annotations import load_bboxes
from PIL import Image
from skimage.measure import regionprops

# %%
# %matplotlib widget

# %%%%%%%%%%%%%%%%%%%%%%%%%%%%%
# September groundtruth data

# map dataset to masks
map_month_to_zarr = {
    "sep": "masks_20261009_133849.zarr",
    "aug": "masks_20261009_135707.zarr",
}

MONTH = "aug"

DATA_DIR = Path(f"/Users/sofia/swc/CrabLabels/{MONTH}2023-full")
IMAGES_DIR = DATA_DIR / "frames"
ANNOTATIONS_FILE = DATA_DIR / "annotations" / "VIA_JSON_combined_coco_gen.json"

LABEL_NAME = "crab"

MASKS_ZARR = DATA_DIR / "annotations" / map_month_to_zarr[MONTH]


# %%%%%%%%%%%%%%%%%%%%%%%%%%
# Helpers


def _read_image(path):
    """Read an image file as a numpy array of shape (H, W, C)."""
    return np.array(Image.open(path))


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
# Load ground truth images as lazy dask array
# Build the list of image files from the dataset's image_id -> filename
# map, so that the images are aligned with the image_id coordinate
images_dir = Path(ds_bboxes.attrs["images_directories"])
map_id_to_filename = ds_bboxes.attrs["map_image_id_to_filename"]
img_paths = [
    images_dir / map_id_to_filename[img_id]
    for img_id in ds_bboxes.image_id.values
]

# assume all images have the same shape and dtype as the first one
sample = _read_image(img_paths[0])  # H, W, C
images = da.stack(
    [
        da.from_delayed(
            dask.delayed(_read_image)(p),
            shape=sample.shape,
            dtype=sample.dtype,
        )
        for p in img_paths
    ]
)  # (image_id, H, W, C), one chunk per image
print(images.shape)

# add as data variable
ds_bboxes["images"] = xr.DataArray(
    data=images,
    dims=["image_id", "img_h", "img_w", "rgb"],
)


# %%%%%%%%%%%%%%%%%%%%%%%
# Load masks
# ID-encoded masks (uint16): 0 = background, k + 1 = the mask
# for the box at position k along `id`. Set the store name to the output of
# scripts/generate_masks_from_bboxes.py
# (older stores are int16, so we cast for consistency)
mask_da_array = da.from_zarr(MASKS_ZARR, mode="r").astype(
    np.uint16
)  # (image_id, H, W)

# check frames in the zarr store are aligned with the image_id coordinate
mask_image_ids = zarr.open(MASKS_ZARR, mode="r").attrs.get("image_ids")
if mask_image_ids is not None:
    assert mask_image_ids == ds_bboxes.image_id.values.tolist()


# %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
# Add masks as label images, with a map from id to label
# (same layout as the output of `mask-tracked-crabs`)
ds_bboxes["labels"] = xr.DataArray(
    data=mask_da_array,
    dims=["image_id", "img_h", "img_w"],
)
# label k + 1 is the box at position k along `id`
ds_bboxes["label_of"] = xr.DataArray(
    data=np.arange(1, ds_bboxes.sizes["id"] + 1, dtype=np.uint16),
    dims=["id"],
)

# %%%%%%%%%%%%%%%%%%%%%%%
# Plot single mask and all masks in frame

# Show single mask for id=10 in frame 40
image_id = 40
id = 10
label = ds_bboxes.label_of.sel(id=id)
single_mask = ds_bboxes.labels.sel(image_id=image_id) == label

fig, ax = plt.subplots(1, 1)

# image
ax.imshow(ds_bboxes.images.sel(image_id=image_id))

# bbox centre
ax.scatter(
    ds_bboxes.position.sel(image_id=image_id, id=id, space="x"),
    ds_bboxes.position.sel(image_id=image_id, id=id, space="y"),
    15,
    marker="x",
    color="r",
)
# single mask by image_id and id
ax.imshow(
    single_mask,
    cmap="Reds",
    alpha=single_mask.astype(float) * 0.5,
)
ax.contour(single_mask, levels=[0.5], colors="red", linewidths=0.5)

# plot all masks in one frame (one color per ID)
label_frame = ds_bboxes.labels.sel(image_id=image_id).values
ax.imshow(
    label_frame,
    cmap="turbo",
    alpha=(label_frame > 0) * 0.5,
)

# %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
# Visualise frames and masks in napari viewer
mask_array = ds_bboxes.labels.data
viewer = napari.Viewer()

# both layers are lazy, so napari only loads the frame displayed
viewer.add_image(ds_bboxes.images.data, name="frames", rgb=True)
viewer.add_labels(mask_array, name=f"{LABEL_NAME} masks")

# %%
# build ellipses and axis lines for napari Shapes layers
all_ellipses = []
all_major_axes = []
all_minor_axes = []
for frame_idx in range(mask_array.shape[0]):
    # masks are already ID-encoded, so each ID is one region
    label_frame = np.asarray(mask_array[frame_idx])
    ellipse_corners, major_axes, minor_axes = ellipses_from_labels(label_frame)
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

# %%
