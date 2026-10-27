"""A notebook to visualise masks in annotations dataset."""

# %%
from pathlib import Path

import dask
import dask.array as da
import matplotlib.pyplot as plt
import napari
import numpy as np
import xarray as xr
from matplotlib.patches import Ellipse
from PIL import Image

# %%
# %matplotlib widget

# %%%%%%%%%%%%%%%%%%%%%%%%%%%%%
# Groundtruth data

# map dataset to the output store of scripts/generate_masks_from_bboxes.py
# (stores named masks_<timestamp>.zarr, written before the bboxes and mask
# parameters were added, cannot be read here; re-run the script)
map_month_to_zarr = {
    "sep": "bboxes_and_masks_20261026_123937.zarr",
    "aug": "bboxes_and_masks_20261026_125630.zarr",
}


# Select input data (sept/aug)
MONTH = "aug"
DATA_DIR = Path.home() / Path(
    f"swc/project_crabs/data/CrabLabels/{MONTH}2023-full"
)
ZARR_STORE = DATA_DIR / "annotations" / map_month_to_zarr[MONTH]


# %%%%%%%%%%%%%%%%%%%%%%%%%%
# Helpers


def _read_image(path):
    """Read an image file as a numpy array of shape (H, W, C)."""
    return np.array(Image.open(path))


def ellipse_shapes_from_ds(ds):
    """Return napari ellipses, major axes and minor axes for every mask.

    In napari, the ellipse shape is represented by the 4 corners of its
    oriented bounding rectangle. The major and minor axes are represented
    by their 2 endpoints, which have coordinates (image index, y, x).

    Returns
    -------
    For N masks:
    - ellipse_corners : (N, 4, 3) array
    - major_axes : (N, 2, 3) array
    - minor_axes : (N, 2, 3) array

    The trailing 3 dimension corresponds to the coordinates
    (image index, y, x).

    """
    # Get ellipses data from dataset

    # (image_id, id) positions of masks
    image_idcs, mask_idcs = np.nonzero(ds.mask_area.notnull().values)

    # mask centroid
    cx = ds.mask_centroid.sel(space="x").values[image_idcs, mask_idcs]
    cy = ds.mask_centroid.sel(space="y").values[image_idcs, mask_idcs]
    c = np.stack([cy, cx], axis=-1)

    # semi-axes lengths
    major_semi_length = (
        ds.ellipse_major_axis_length.values[image_idcs, mask_idcs][:, None] / 2
    )
    minor_semi_length = (
        ds.ellipse_minor_axis_length.values[image_idcs, mask_idcs][:, None] / 2
    )

    # ellipse orientation wrt x axis image coords
    theta = ds.ellipse_major_axis_theta.values[image_idcs, mask_idcs]

    # axes unit vectors in (y, x) space
    major_unit_vector = np.stack([np.sin(theta), np.cos(theta)], axis=-1)
    minor_unit_vector = np.stack([np.cos(theta), -np.sin(theta)], axis=-1)

    # semi-axis vectors in (y, x)
    major_semi_vector = major_semi_length * major_unit_vector
    minor_semi_vector = minor_semi_length * minor_unit_vector

    # corners of the ellipse's oriented bounding box
    corners = np.stack(
        [
            c + major_semi_vector + minor_semi_vector,
            c + major_semi_vector - minor_semi_vector,
            c - major_semi_vector - minor_semi_vector,
            c - major_semi_vector + minor_semi_vector,
        ],
        axis=1,
    )  # (N, 4, 2)

    # axes segments
    major_axis_endpoints = np.stack(
        [c - major_semi_vector, c + major_semi_vector], axis=1
    )  # (N, 2, 2)
    minor_axis_endpoints = np.stack(
        [c - minor_semi_vector, c + minor_semi_vector], axis=1
    )  # (N, 2, 2)

    def add_image_idx(points):
        """Add image index coordinate to the input `points` array.

        The input `points` array is of shape (N, n_points, 2),
        where N is the number of masks, n_points is the number
        of points required to define the shape (n_points=4 for
        corners, n_points=2 for axes segments), and 2 are the
        spatial dimensions (y,x).
        """
        # Broadcast image_idcs (N,) to (N, n_points, 1) to match
        # the first two dimensions of the points array.
        image_idcs_b = np.broadcast_to(
            image_idcs[:, None, None],  # (N,) --> (N, 1, 1)
            (*points.shape[:2], 1),  # (N, n_points, 1)
        )  # (N, n_points, 1)

        # Concatenate image_idcs along the last axis (coordinates)
        # (N, n_points, 1) + (N, n_points, 2) --> (N, n_points, 3)
        return np.concatenate([image_idcs_b, points], axis=-1)

    return (
        add_image_idx(corners),
        add_image_idx(major_axis_endpoints),
        add_image_idx(minor_axis_endpoints),
    )


# %%%%%%%%%%%%%%%%%%%%%%
# Read bboxes, masks and geometric parameters from zarr store
# - labels: ID-encoded masks (uint16), 0 = background
# - label_of: label k + 1 is the mask for the box at position k along `id`
# - mask_* and ellipse_*: parameters per mask, on (image_id, id)
ds = xr.open_zarr(ZARR_STORE)

print(ds.data_vars)

# %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
# Load ground truth images into dataset
# as a lazy dask array
# (image files are aligned with the image_id coordinate)

# zarr stores attrs as JSON, so the image_id keys read back as str,
# which we need to cast to int here
map_id_to_filename = {
    int(k): v for k, v in ds.attrs["map_image_id_to_filename"].items()
}
images_dir = Path(ds.attrs["images_directories"])
img_paths = [
    images_dir / map_id_to_filename[img_id] for img_id in ds.image_id.values
]

# assume all images have the same shape and dtype as the first one
sample = _read_image(img_paths[0])  # H, W, C
# define lazy reader
read_lazy = dask.delayed(_read_image)
images = da.stack(
    [
        da.from_delayed(
            read_lazy(path),
            shape=sample.shape,
            dtype=sample.dtype,
        )
        for path in img_paths
    ]  # list of dask arrays
)  # (image_id, H, W, C), one chunk per image
print(images.shape)

# add image array as data variable
ds["images"] = xr.DataArray(
    data=images,
    dims=["image_id", "img_h", "img_w", "rgb"],
)

# %%%%%%%%%%%%%%%%%%%%%%
# Get bbox, mask and ellipse of one box
image_id = 40
id = 10  # annotation id
box = ds.sel(image_id=image_id, id=id)

print(f"bbox centre (x, y): {box.position.values}")
print(f"bbox (width, height): {box.shape.values}")
print(f"mask centroid (x, y): {box.mask_centroid.values}")
print(f"mask area (pixels): {box.mask_area.values}")
print(f"ellipse major axis (pixels): {box.ellipse_major_axis_length.values}")
print(
    "ellipse major axis theta (radians): "
    f"{box.ellipse_major_axis_theta.values}"
)



# %%%%%%%%%%%%%%%%%%%%%%%
# Plot single mask and all masks in frame

single_mask = ds.labels.sel(image_id=image_id) == box.label_of


# image
fig, ax = plt.subplots(1, 1)
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

# add contour
ax.contour(single_mask, levels=[0.5], colors="red", linewidths=0.5)

# add ellipse 
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
viewer = napari.Viewer()

# both layers are lazy, so napari only loads the frame displayed

# add frames
viewer.add_image(ds.images.data, name="frames", rgb=True)

# add masks
mask_array = ds.labels.data
viewer.add_labels(mask_array, name="masks")

# %%
# add ellipses and axis lines as napari Shapes layers,
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
