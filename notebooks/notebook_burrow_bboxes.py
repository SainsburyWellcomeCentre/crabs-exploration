"""A notebook exploring how to programmatically extract burrow prompts.

The prompts are bboxes that are passed to SAM3 for segmenting burrows.
"""

# %%%%%%%%%%%%%%%%%%%%%%%%%%%
# Imports
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import xarray as xr
from scipy.ndimage import find_objects, gaussian_filter, label
import numpy as np
from scipy.ndimage import gaussian_filter
from skimage.feature import peak_local_max

# Hide attributes globally
xr.set_options(
    display_expand_attrs=False,
    display_style="html",  # "text" for readibility in dark mode?
)

# %%
# pip install ipympl first for interactive plots
%matplotlib widget

# %%%%%%%%%%%%%%%%
# Input data

data_dir = Path().home() / "swc" / "project_crabs" / "data"/ "CrabTracks"
crabs_zarr_dataset = data_dir / "CrabTracks-slurm2478780-2478861-2489356.zarr"

image_w = 4096
image_h = 2160


# %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
# Read dataset as an xarray datatree

dt = xr.open_datatree(
    crabs_zarr_dataset,
    engine="zarr",
    chunks={},
)

print(dt)


# %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
# Flatten trajectories in one video 

# .to_dataset(): makes a copy
# .ds(): returns a view, changes propagate to tree
ds_video = dt["10.08.2023-02-Right"].ds

# prepare data
# flatten to just time and space coords
position_x = ds_video.position.sel(space="x").values
position_y = ds_video.position.sel(space="y").values
slc_non_nan = ~np.isnan(position_x) & ~np.isnan(position_y)
points_x = position_x[slc_non_nan]  
points_y = position_y[slc_non_nan]

# free original arrays
del position_x, position_y


# %%
# Plot 2D histogram
# # define bins
# bin_size_pixels = 5  
# n_bins_x = round(image_w / bin_size_pixels)
# n_bins_y = round(image_h / bin_size_pixels)

# # plot for all clips in video
# fig, ax = plt.subplots()
# h, _, _, img = ax.hist2d(
#     points_x,
#     points_y,
#     bins=[n_bins_x, n_bins_y],
#     cmap="viridis",
#     # cap count at 1min
#     vmax=ds_video.fps*60, # frames in 1min # np.percentile(h.flatten(), 99)
# )
# fig.colorbar(img, ax=ax, label="count")
# ax.invert_yaxis()
# ax.set_xlabel("x (pixels)")
# ax.set_ylabel("y (pixels)")
# ax.set_aspect("equal")
# ax.set_title(ds_video.video_id)


# %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
# Compute 2D histogram, 
# 1. Discretize data into bins size 10x10 pixels
# 2. Keep bin counts > 95th percentile
# 3. Apply Gaussian filter

# prepare plots
# fig, axes = plt.subplots(3, 2, figsize=(16, 8))
# fig.suptitle(f"Node detection pipeline {ds_video.video_id}", fontsize=14)

# %%
# Compute histogram
bin_size_pixels = 5
n_bins_x = round(image_w / bin_size_pixels)
n_bins_y = round(image_h / bin_size_pixels)

counts, xedges, yedges = np.histogram2d(
    points_x,
    points_y,
    bins=[n_bins_x, n_bins_y],  
    range=[[0, image_w], [0, image_h]],
    # define range per video?
)

# plot histogram (max color set to X percentile)
# ax = axes[0, 0]
fig, ax = plt.subplots(1,1)
img = ax.imshow(
    counts.T,
    origin="upper",
    aspect="equal",
    cmap="viridis",
    vmax=np.percentile(counts, 99)
)
fig.colorbar(img, ax=ax, label="count")
ax.set_aspect("equal")
ax.set_title(f"Histogram {ds_video.video_id}")
ax.set_xlabel("x (pixels)")
ax.set_ylabel("y (pixels)")


# %%
# Set bins with low counts to zero
min_count = np.percentile(counts, 99)
counts_filtered = np.where(counts >= min_count, counts, 0)

fig, ax = plt.subplots(1,1)
img = ax.imshow(
    counts_filtered.T > 0,
    origin="upper",
    aspect="equal",
    cmap="viridis",
    # vmax=min_count, 
)
fig.colorbar(img, ax=ax, label="retained")
ax.set_aspect("equal")
ax.set_title(f"Retained bins {ds_video.video_id}")
ax.set_xlabel("x (pixels)")
ax.set_ylabel("y (pixels)")

# %%
# Apply transform to filtered bins ---- why?
# log or power
log_counts = np.log1p(counts_filtered)  # counts_filtered ** 0.5 #

fig, ax = plt.subplots(1,1)
img = ax.imshow(
    log_counts.T,
    origin="upper",
    aspect="equal",
    cmap="Blues",
)
fig.colorbar(img, ax=ax, label="transformed count")
ax.set_aspect("equal")
ax.set_title("Transformed filtered counts")
ax.set_xlabel("x (pixels)")
ax.set_ylabel("y (pixels)")

# %%
# Option 1: Gaussian filter + find peaks
# Apply Gaussian smoothing
# log_counts = counts
sigma = 2.5  # <---------- kernel radius = 4*sigma ~ blob size?
smoothed = gaussian_filter(log_counts, sigma=sigma)

fig, ax = plt.subplots(1,1)
ax.imshow(
    smoothed.T,
    origin="upper",
    aspect="equal",
    cmap="Blues",
    extent=[0, image_w, image_h, 0],
    # vmax=np.percentile(log_counts.flatten(),99.99),
)
ax.set_title(f"3. Smoothed (gaussian, sigma={sigma})")
ax.set_xlabel("x (pixels)")
ax.set_ylabel("y (pixels)")

# %%
# Find peaks
peaks_col_row = peak_local_max(
    smoothed,
    min_distance=10,  # <----------
    threshold_rel=0.6,  # min intensity of peaks <----------
)
x_bin_centers = (xedges[:-1] + xedges[1:]) / 2
y_bin_centers = (yedges[:-1] + yedges[1:]) / 2
node_x = x_bin_centers[peaks_col_row[:, 0]]
node_y = y_bin_centers[peaks_col_row[:, 1]]


# plot peaks on smoothed data
fig, ax = plt.subplots(1,1)
ax.imshow(
    smoothed.T,
    origin="upper",
    aspect="equal",
    cmap="Blues",
    extent=[0, image_w, image_h, 0],
    # left, right, bottom, top
    # map image array idcs to coords
)
ax.scatter(
    node_x,
    node_y,
    s=25,
    c="red",
    marker="x",
    linewidths=0.5,
    label=f"{len(peaks_col_row)} peaks",
)
ax.set_title(f"Detected peaks {ds_video.video_id}")
ax.set_xlabel("x (pixels)")
ax.set_ylabel("y (pixels)")
ax.legend(fontsize=8)

fig, ax = plt.subplots(1,1)
ax.imshow(
    counts.T, #<---------
    origin="upper",
    aspect="equal",
    cmap="Blues",
    extent=[0, image_w, image_h, 0],
    # left, right, bottom, top
    # map image array idcs to coords
    vmax = np.percentile(counts, 95)
)
ax.scatter(
    node_x,
    node_y,
    s=25,
    c="red",
    marker="x",
    linewidths=0.5,
    label=f"{len(peaks_col_row)} peaks",
)
ax.set_title(f"Detected peaks {ds_video.video_id}")
ax.set_xlabel("x (pixels)")
ax.set_ylabel("y (pixels)")
ax.legend(fontsize=8)

# %%%%%%%%%%%%%%%%%%%%
# Option 2: Use blob_doh instead
from skimage.feature import blob_log, blob_doh

blobs = blob_doh(log_counts, min_sigma=3, max_sigma=10, num_sigma=5, threshold_rel=0.5,)
# blobs is an array of [y, x, sigma] for each detected blob

fig, ax = plt.subplots(1,1)
ax.imshow(
    log_counts.T, #<---------
    origin="upper",
    aspect="equal",
    cmap="Blues",
    # extent=[0, image_w, image_h, 0],
    # left, right, bottom, top
    # map image array idcs to coords
)
ax.scatter(
    blobs[:, 0],
    blobs[:, 1],
    s=25,
    c="red",
    marker="x",
    linewidths=0.5,
    label=f"{len(blobs)} peaks",
)
ax.set_title(f"Detected peaks {ds_video.video_id}")
ax.set_xlabel("x (pixels)")
ax.set_ylabel("y (pixels)")
ax.legend(fontsize=8)

fig, ax = plt.subplots(1,1)
ax.imshow(
    counts.T, #<---------
    origin="upper",
    aspect="equal",
    cmap="Blues",
    # extent=[0, image_w, image_h, 0],
    # left, right, bottom, top
    # map image array idcs to coords
    vmax = np.percentile(counts, 95)
)
ax.scatter(
    blobs[:, 0],
    blobs[:, 1],
    s=25,
    c="red",
    marker="x",
    linewidths=0.5,
    label=f"{len(blobs)} peaks",
)
ax.set_title(f"Detected peaks {ds_video.video_id}")
ax.set_xlabel("x (pixels)")
ax.set_ylabel("y (pixels)")
ax.legend(fontsize=8)



# %%
