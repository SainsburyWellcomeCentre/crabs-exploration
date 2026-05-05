"""A notebook to visualise masks in annotations dataset."""

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
# %matplotlib widget

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

# add as attribute
ds_bboxes.attrs["image_array"] = image_array


# %%%%%%%%%%%%%%%%%%%%%%%
# Load masks
zarr_root = zarr.open(
    DATA_DIR / f"{LABEL_NAME} masks.zarr",
    mode="r",
)

mask_da_array = da.from_zarr(zarr_root["masks"])


# %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
# Load as 4D boolean dask array with id dim
# Each slice [image_id, id] is a boolean mask for that object in that frame
# CONS: can be slow to compute
n_ids = ds_bboxes.sizes["id"]

# for one task per chunk:
def _apply_one_hot_encoding_for_IDs(block, n_ids):
    """Transform a block of shape (chunk_frames, H, W) 
    into a boolean array of shape (chunk_frames, n_ids, H, W).

    This is most efficient if chunk_frames=1.
    """
    labels = np.arange(1, n_ids + 1)[None, :, None, None] # (1, n_ids, 1, 1)
    # compare every pixel against every label simultaneously
    # via broadcasting (i.e. for each pixel, we have a 1-hot encoding
    # vector indicate which label/id it is)
    return block[:, None, :, :] == labels


# Apply expand labels chunk by chunk
# we define one chunk per image
mask_da_array = mask_da_array.rechunk({0: 1}) 
mask_4d = mask_da_array.map_blocks(
    _apply_one_hot_encoding_for_IDs,
    n_ids=n_ids,
    new_axis=1, # position of new_axis id
    chunks=( # sizes of output chunks (if diff from input)
        mask_da_array.chunks[0], # same as input chunk axis=0
        (n_ids,),   # ids
        mask_da_array.chunks[1], # same as input chunk axis=1
        mask_da_array.chunks[2], # same as input chunk axis=2
    ),
    dtype=bool,
)  # (image_id, id, img_h, img_w)

ds_bboxes["masks_bool"] = xr.DataArray(
    data=mask_4d,
    dims=["image_id", "id", "img_h", "img_w"],
)
# %%%%%%%%%%%%%%%%%%%%%%%
# Plot single mask and all masks in frame

# Show single mask for id=10 in frame 40
image_id = 40
id = 10
single_mask = ds_bboxes.masks_bool.sel(image_id=image_id, id=id)

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
ax.imshow(
    single_mask,
    cmap="Reds",
    alpha=single_mask.astype(float) * 0.5,
)
ax.contour(single_mask, levels=[0.5], colors="red", linewidths=0.5)

# plot all masks in one frame (boolean masks, all same color!)
all_masks = ds_bboxes.masks_bool.sel(image_id=image_id).any(dim="id")
ax.imshow(
    all_masks,
    cmap="turbo",
    alpha=all_masks.astype(float) * 0.5,
)

# %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
# Visualise frames and masks in napari viewer
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
