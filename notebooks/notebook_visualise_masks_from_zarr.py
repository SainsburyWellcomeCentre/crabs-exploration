"""A notebook to visualise masks in annotations dataset."""

# %%
import sys
from pathlib import Path

import dask
import dask.array as da
import matplotlib.pyplot as plt
import napari
import numpy as np
import xarray as xr
from matplotlib.patches import Ellipse
from PIL import Image

# make scripts/ importable, whether this runs from notebooks/ or the repo root
REPO_ROOT = next(
    p for p in [Path.cwd(), *Path.cwd().parents] if (p / "scripts").is_dir()
)
sys.path.insert(0, str(REPO_ROOT))
from scripts.generate_masks_from_bboxes import get_image_paths  # noqa: E402

# %%
# %matplotlib widget

# %%%%%%%%%%%%%%%%%%%%%%%%%%%%%
# Groundtruth data

# map dataset to the output store of scripts/generate_masks_from_bboxes.py
# (stores named masks_<timestamp>.zarr, written before the bboxes and mask
# parameters were added, cannot be read here; re-run the script)
map_month_to_zarr = {
    "sep": "bboxes_and_masks_<timestamp>.zarr",
    "aug": "bboxes_and_masks_<timestamp>.zarr",
}

MONTH = "aug"

DATA_DIR = Path(f"/Users/sofia/swc/CrabLabels/{MONTH}2023-full")

LABEL_NAME = "crab"

ZARR_STORE = DATA_DIR / "annotations" / map_month_to_zarr[MONTH]


# %%%%%%%%%%%%%%%%%%%%%%%%%%
# Helpers


def _read_image(path):
    """Read an image file as a numpy array of shape (H, W, C)."""
    return np.array(Image.open(path))


def ellipse_shapes_from_ds(ds):
    """Return napari ellipses, major axes and minor axes for every mask.

    Each ellipse is represented by the 4 corners of its oriented bounding
    rectangle (for napari's built-in ``"ellipse"`` shape type), and each
    axis by its 2 endpoints. Points are in (frame index, y, x) order.

    Returns
    -------
    ellipse_corners : (N, 4, 3) array
    major_axes : (N, 2, 3) array
    minor_axes : (N, 2, 3) array

    """
    f, k = np.nonzero(ds.mask_area.notnull().values)  # frame, id index
    cx = ds.mask_centroid.sel(space="x").values[f, k]
    cy = ds.mask_centroid.sel(space="y").values[f, k]
    a = ds.ellipse_major_axis_length.values[f, k][:, None] / 2
    b = ds.ellipse_minor_axis_length.values[f, k][:, None] / 2
    theta = ds.ellipse_major_axis_theta.values[f, k]

    # unit vectors in (y, x): major axis at theta from +x, minor at 90 deg
    u = np.stack([np.sin(theta), np.cos(theta)], axis=-1)
    v = np.stack([np.cos(theta), -np.sin(theta)], axis=-1)
    c = np.stack([cy, cx], axis=-1)

    corners = np.stack(
        [
            c + a * u + b * v,
            c + a * u - b * v,
            c - a * u - b * v,
            c - a * u + b * v,
        ],
        axis=1,
    )  # (N, 4, 2)
    major = np.stack([c - a * u, c + a * u], axis=1)  # (N, 2, 2)
    minor = np.stack([c - b * v, c + b * v], axis=1)  # (N, 2, 2)

    def with_frame(points):
        frame = np.broadcast_to(f[:, None, None], (*points.shape[:2], 1))
        return np.concatenate([frame, points], axis=-1)

    return with_frame(corners), with_frame(major), with_frame(minor)


# %%%%%%%%%%%%%%%%%%%%%%
# Read bboxes, masks and mask parameters from one store
# - labels: ID-encoded masks (uint16), 0 = background
# - label_of: label k + 1 is the mask for the box at position k along `id`
# - mask_* and ellipse_*: parameters per mask, on (image_id, id)
ds = xr.open_zarr(ZARR_STORE)

# %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
# Load ground truth images as lazy dask array
# image files aligned with the image_id coordinate, as in the script
img_paths = get_image_paths(ds)

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
ds["images"] = xr.DataArray(
    data=images,
    dims=["image_id", "img_h", "img_w", "rgb"],
)

# %%%%%%%%%%%%%%%%%%%%%%
# Bbox, mask and ellipse of one box, all at the same (image_id, id)
image_id = 40
id = 10
box = ds.sel(image_id=image_id, id=id)

print(box.position.values)  # bbox centre (x, y)
print(box.shape.values)  # bbox width, height
print(box.mask_centroid.values)  # mask centroid (x, y)
print(box.mask_area.values)  # mask area, in pixels
print(box.ellipse_major_axis_length.values)  # full length, in pixels
print(box.ellipse_major_axis_theta.values)  # radians, from +x towards +y
single_mask = ds.labels.sel(image_id=image_id) == box.label_of

# %%%%%%%%%%%%%%%%%%%%%%%
# Plot single mask and all masks in frame

fig, ax = plt.subplots(1, 1)

# image
ax.imshow(ds.images.sel(image_id=image_id))

# bbox centre
ax.scatter(
    ds.position.sel(image_id=image_id, id=id, space="x"),
    ds.position.sel(image_id=image_id, id=id, space="y"),
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

# ellipse fitted to the single mask
ax.add_patch(
    Ellipse(
        xy=box.mask_centroid.values,  # (x, y)
        width=float(box.ellipse_major_axis_length),
        height=float(box.ellipse_minor_axis_length),
        angle=np.degrees(float(box.ellipse_major_axis_theta)),
        fill=False,
        edgecolor="yellow",
    )
)

# plot all masks in one frame (one color per ID)
label_frame = ds.labels.sel(image_id=image_id).values
ax.imshow(
    label_frame,
    cmap="turbo",
    alpha=(label_frame > 0) * 0.5,
)

# %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
# Visualise frames and masks in napari viewer
mask_array = ds.labels.data
viewer = napari.Viewer()

# both layers are lazy, so napari only loads the frame displayed
viewer.add_image(ds.images.data, name="frames", rgb=True)
viewer.add_labels(mask_array, name=f"{LABEL_NAME} masks")

# %%
# ellipses and axis lines for napari Shapes layers,
# from the stored ellipse parameters
all_ellipses, all_major_axes, all_minor_axes = ellipse_shapes_from_ds(ds)

viewer.add_shapes(
    list(all_ellipses),
    shape_type="ellipse",
    edge_color="yellow",
    edge_width=4,
    face_color="transparent",
    name="ellipses",
)
viewer.add_shapes(
    list(all_minor_axes),
    shape_type="line",
    edge_color="green",
    edge_width=4,
    name="minor axes",
)
viewer.add_shapes(
    list(all_major_axes),
    shape_type="line",
    edge_color="red",
    edge_width=4,
    name="major axes",
)

# %%
