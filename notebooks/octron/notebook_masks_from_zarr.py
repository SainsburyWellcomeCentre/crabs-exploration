# %%
from pathlib import Path

import dask.array as da
import matplotlib.pyplot as plt
import napari
import numpy as np
import sparse
import xarray as xr
import zarr
from ethology.io.annotations import load_bboxes
from PIL import Image
# %%
%matplotlib widget

# %%%%%%%%%%%%%%%%%%%%%%%%%%%%%
# September groundtruth data
DATA_DIR = Path("/Users/sofia/swc/CrabLabels/sep2023-full")
IMAGES_DIR = DATA_DIR / "frames"
ANNOTATIONS_FILE = DATA_DIR / "annotations" / "VIA_JSON_combined_coco_gen.json"

LABEL_NAME = "crab"

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


# %%%%%%%%%%%%%%%%%%%%%%%
# Load masks
output_masks_dir = Path(
    "/Users/sofia/arc/project_Zoo_crabs/crabs-exploration/output/sep2023-full"
)

zarr_root = zarr.open(
    output_masks_dir / f"{LABEL_NAME} masks.zarr",
    mode="r",
)

mask_da_array = da.from_zarr(zarr_root["masks"])


# %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
# How to add masks to bboxes ds with aligned ids

# What I would like is something like
# ds_bboxes.masks.sel(image_id=0) ---> returns masks for that frame
# ds_bboxes.masks.sel(image_id=0, id=3) ---> returns mask for that frame and id=3


# %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
# Option 1: as dask-arrays
# (but we lose alignment with ID) --- except if ID matches label?
ds_bboxes.assign_coords(
    {
        "img_h": np.arange(mask_da_array.shape[1]),
        "img_w": np.arange(mask_da_array.shape[2]),
    }
)

ds_bboxes["masks"] = xr.DataArray(
    data=mask_da_array,
    dims=["image_id", "img_h", "img_w"],
)

# %%
# Example usage

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
    color="r",
)
# red contour around corresponding mask
# NOTE id + 1, since label=0 in the mask is background
single_mask = (ds_bboxes.masks.sel(image_id=image_id) == id + 1)
ax.imshow(
    single_mask,
    cmap="Reds",
    alpha=single_mask.astype(float)*0.5,
)
ax.contour(single_mask, levels=[0.5], colors="red", linewidths=0.5)
# all masks
all_masks_one_img = ds_bboxes.masks.sel(image_id=image_id)
ax.imshow(
    all_masks_one_img,
    cmap="turbo",
    alpha=(all_masks_one_img > 0).astype(float)*0.5,
)

# %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
# Option 2: store sparse array objects
# PRO: we store less data, a compact representation of the mask
# CON: we lose vectorised ops and cannot serialise (unless we do .todense first)

# Initialise array
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
# Example usage

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
    color="r",
)
# red contour around corresponding mask
single_mask_dense = (
    ds_bboxes.masks_sparse.isel(image_id=image_id, id=id).item().todense()
)
ax.imshow(
    single_mask_dense,
    cmap="Reds",
    alpha=(single_mask_dense > 0).astype(float)*0.5,
)
ax.contour(single_mask_dense, levels=[0.5], colors="red", linewidths=0.5)

# all masks
all_masks_one_img = sparse.stack(
    ds_bboxes.masks_sparse.sel(image_id=image_id).dropna(dim='id').values
).todense().max(axis=0)

ax.imshow(
    all_masks_one_img,
    cmap="turbo",
    alpha=(all_masks_one_img > 0).astype(float)*0.5,
)


# %%%%%%%%%%
# Option 3: store dask array objects?
# CON: with objects we lose vectorized ops and serialising
# construction is instant but access is slow (opposite to sparse)
# PRO: we can sel by id?

mask_da_obj = np.empty(
    (ds_bboxes.sizes["image_id"], ds_bboxes.sizes["id"]),
    dtype=object,
)

for image_id in range(mask_da_array.shape[0]):
    mask_frame = mask_da_array[image_id]  # still a dask array, not computed
    for l_i, lbl in enumerate(range(1, ds_bboxes.sizes["id"] + 1)):
        mask_da_obj[image_id, l_i] = (mask_frame == lbl)  # dask array (img_h, img_w)

ds_bboxes["masks_da_obj"] = xr.DataArray(
    data=mask_da_obj,
    dims=["image_id", "id"],
)

# %%
# Usage
# single mask — returns a dask array, call .compute() to get numpy
single_mask = ds_bboxes.masks_da_obj.sel(image_id=40, id=10).item().compute()

# all masks in one frame
all_masks = da.stack(
    ds_bboxes.masks_da_obj.isel(image_id=40).dropna(dim='id').values
).compute().max(axis=0)

# %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
# Option 4: 4D boolean dask array with id dim
# Each slice [image_id, id] is a boolean mask for that object in that frame
# CONS: slow to compute
n_ids = ds_bboxes.sizes["id"]

# for one task per chunk:
def _expand_labels(block, n_ids):
    """(chunk_frames, H, W) -> (chunk_frames, n_ids, H, W) boolean."""
    labels = np.arange(1, n_ids + 1)[None, :, None, None]
    return block[:, None, :, :] == labels


mask_4d = mask_da_array.map_blocks(
    _expand_labels,
    n_ids=n_ids,
    new_axis=1,
    chunks=(
        mask_da_array.chunks[0],
        (n_ids,),
        mask_da_array.chunks[1],
        mask_da_array.chunks[2],
    ),
    dtype=bool,
)  # (image_id, id, img_h, img_w)

ds_bboxes["masks_bool"] = xr.DataArray(
    data=mask_4d,
    dims=["image_id", "id", "img_h", "img_w"],
)
# %%
# Usage

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
    color="r",
)
# single mask by image_id and id
single_mask = ds_bboxes.masks_bool.sel(image_id=image_id, id=id)
ax.imshow(
    single_mask,
    cmap="Reds",
    alpha=single_mask.astype(float) * 0.5,
)
ax.contour(single_mask, levels=[0.5], colors="red", linewidths=0.5)

# all masks in one frame (boolean masks, all same color!)
all_masks = ds_bboxes.masks_bool.sel(image_id=image_id).any(dim="id")
ax.imshow(
    all_masks,
    cmap="turbo",
    alpha=all_masks.astype(float) * 0.5,
)

# %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
# Load frames and masks in napari viewer
mask_array = mask_da_array
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
