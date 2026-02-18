"""Demo notebook for working with crab dataset.



Useful references:
- https://docs.xarray.dev/en/latest/api/datatree.html

"""

# %%
import os
import random
from collections import defaultdict
from pathlib import Path

import matplotlib.pyplot as plt
import psutil
import xarray as xr
import zarr
from movement.io import load_bboxes
from movement.plots import plot_centroid_trajectory
import numpy as np

# Hide attributes globally
xr.set_options(
    display_expand_attrs=False, 
    display_style='html',  # for readibility in dark mode
)

# %%
%matplotlib widget

# %%%%%%%%%%%%%%%%
# Input data

crabs_zarr_dataset = "/ceph/zoo/processed/CrabField/ramalhete_2023/CrabTracks-slurm2412462-slurm2423692.zarr"

data_vars_order = [
    "position",
    "shape",
    "confidence",
    "escape_state",
]

# %%%%%%%%%%%%%
# Read dataset as an xarray datatree

dt = xr.open_datatree(
    crabs_zarr_dataset,
    engine="zarr",
    chunks={},
)

print(dt)

# %%%%%%%%%%%%%%%
# Inspect datatree
print(f"Depth: {dt.depth}")

# Number of groups (i.e. videos)
print(f"Number of groups: {len(dt.groups)}")

# Print clips per video
for ds_video in dt.leaves:
    print(f"{ds_video.path}: {len(ds_video.clip_id)} clips")





# %%%%%%%%%%%%%%%
# Inspect a single video
ds_video = dt["04.09.2023-01-Right"]
print(ds_video) # shows dask arrays unloaded

# Load coordinates only
# (any coord will do, we choose clip_id)
# .load(): applies in-place,
# .compute(): returns a new object 
ds_video.coords['clip_id'].load()
print(ds_video) 


# Load all data
# Compare memory before and after loading dask arrays into memory.
# Check memory before
# process = psutil.Process(os.getpid())
# mem_before = process.memory_info().rss / 1_000_000_000
# print(f"Memory before: {mem_before:.2f} GB")

# # Load the data
# ds = dt["04.09.2023-01-Right"]
# ds_loaded = ds.load()

# # Check memory after
# mem_after = process.memory_info().rss / 1_000_000_000
# print(f"Memory after: {mem_after:.2f} GB")

# %%%%%%%%%%%%%%%%
# Inspect another video without loading all data
ds_video2 = dt["04.09.2023-02-Right"].to_dataset()
# reorder data vars
ds_video2 = ds_video2[data_vars_order]
print(ds_video2)  # shows dask arrays unloaded
print('------------------------------')

# Load coords only
# NOTE: "time" coord is 0 when clip starts!
ds_video2.coords['clip_id'].load()
print(ds_video2)  # shows dask arrays unloaded, but coords loaded
print('------------------------------')

# Check dataset attributes
print("Dataset attributes:")
print(ds_video2.attrs)
print('------------------------------')

# %%
# Inspect clips in video

# How many clips?
print(f"Video ID: {ds_video2.video_id}")
print(f"Number of clips: {ds_video2.coords['clip_id'].size}")
print(f"Escape types: {np.unique(ds_video2.clip_escape_type.values)}")
print(
    (f"Number of triggered clips: "
    f"{ds_video2.where(ds_video2.clip_escape_type=='triggered', drop=True).coords['clip_id'].size}"))
print(
    (f"Number of spontaneous clips: "
    f"{ds_video2.where(ds_video2.clip_escape_type=='spontaneous', drop=True).coords['clip_id'].size}"))

# Get escape type for loop at index 1
print(ds_video2.isel(clip_id=1).clip_escape_type.values.item())



# %%
# Visualise clips in video as a bar plot
# prepare for plot
list_colors = [plt.get_cmap("tab20c").colors[i] for i in [2,3]]  # 2 colors
escape_colors = {
    "triggered": 'k',
    "spontaneous": 'r',
}
bar_height_plot = 1
bar_widths = (ds_video2.clip_last_frame_0idx.values - ds_video2.clip_first_frame_0idx.values) + 1
bar_edges = ds_video2.clip_first_frame_0idx.values

# plot horizontal bar plot
fig, ax = plt.subplots(1, 1, figsize=(8, 5))
rects = ax.barh(
    y=np.zeros_like(bar_widths),
    width=bar_widths,
    left=bar_edges,
    height=bar_height_plot,
    color=list_colors,
)
# add vertical lines for escape frames
ax.vlines(
    ds_video2.clip_escape_first_frame_0idx.values,
    ymin=-bar_height_plot / 2,
    ymax=bar_height_plot / 2,
    color=[escape_colors[escape_type] for escape_type in ds_video2.clip_escape_type.values],
    linestyle="--",
)
ax.set_xlim(0, ds_video2.clip_last_frame_0idx.values.max())
ax.set_ylim(
    -bar_height_plot / 2,
    bar_height_plot / 2,
)
ax.set_xlabel("frames")
ax.yaxis.set_visible(False)

ax.set_aspect(25_000)
ax.set_title(ds_video2.video_id)

# %%%%%%%%%%%%%
# Compute path length per individual


# %%
# Select one clip
ds_clip = ds_video2.isel(clip_id=0)
# or equivalently: ds_clip = ds_video2.sel(clip_id="Loop00")


# to show stats of confidence values per clip
ds_clip.confidence.mean().compute()
ds_clip.confidence.std().compute()
ds_clip.confidence.min().compute()
ds_clip.confidence.max().compute()
ds_clip.confidence.median(dim=("individuals")).compute()

# for all video
ds_video2.confidence.median(dim=("time", "individuals")).compute()



# %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
# Plot all individuals one clip
# video_name = "04.09.2023-01-Right"

# Select a clip
ds_clip = ds_video2.isel(clip_id=0)
ds_clip.escape_state.load() # to be able to index by escape state 

# Pre-compute individual -> color mapping
colors = plt.cm.tab20.colors
color_map = {
    ind.item(): colors[i % len(colors)]
    for i, ind in enumerate(ds_clip.individuals)
}

# Outbound data (escape_state == 0)
pos_out = ds_clip.position.where(ds_clip.escape_state == 0.0, drop=True)
pos_out_flat = pos_out.stack(flat=("time", "individuals")).dropna("flat")

# color assignment per individual for outbound
ind_labels = pos_out_flat.individuals.values
c_out = np.array([color_map[ind] for ind in ind_labels])

# Inbound data (escape_state == 1)
pos_in = ds_clip.position.where(ds_clip.escape_state == 1.0, drop=True)
pos_in_flat = pos_in.stack(flat=("time", "individuals")).dropna("flat")

# color assignment per individual for inbound
ind_labels = pos_in_flat.individuals.values
c_in = np.array([color_map[ind] for ind in ind_labels])


fig, ax = plt.subplots(1, 1)
ax.scatter(
    pos_out_flat.sel(space="x").values,
    pos_out_flat.sel(space="y").values,
    marker="o",
    c=c_out,
    s=15,
)

ax.scatter(
    pos_in_flat.sel(space="x").values,
    pos_in_flat.sel(space="y").values,
    marker="o",
    c="r",
    s=15,
)

# add a ring around red marker
ax.scatter(
    pos_in_flat.sel(space="x").values,
    pos_in_flat.sel(space="y").values,
    marker="x",
    facecolors=c_in,
    s=1,
)

ax.invert_yaxis()
ax.set_xlabel("x (pixels)")
ax.set_ylabel("y (pixels)")
ax.set_aspect("equal")
ax.set_title(
    f"{ds_clip.clip_id.values.item()} - {ds_clip.clip_escape_type.values.item()}"
)


# %%
# plot all individuals --- much slower bc we loop per individual
# fig, ax = plt.subplots()
# colors = plt.cm.tab20.colors
# for i, ind in enumerate(ds_clip.individuals):
#     # plot outbound
#     plot_centroid_trajectory(
#         ds_clip.position.where(ds_clip.escape_state == 0.0, drop=True),
#         individual=ind,
#         ax=ax,
#         c=colors[i % len(colors)],
#     )
#     # plot inbound
#     plot_centroid_trajectory(
#         ds_clip.position.where(ds_clip.escape_state == 1.0, drop=True),
#         individual=ind,
#         ax=ax,
#         c="r",
#     )
# ax.invert_yaxis()
# ax.set_xlabel("x (pixels)")
# ax.set_ylabel("y (pixels)")
# ax.set_aspect("equal")
# ax.set_title(
#     f"{ds_clip.clip_id.values.item()} - {ds_clip.clip_escape_type.values.item()}"
# )


# %%%%%%%%%
# Plot occupancy for all clips in same video

# prepare data
# flatten to just time and space coords
position_flat = ds_video2.position.stack(
    flat_dim=("clip_id", "time", "individuals")
).dropna("flat_dim")


# plot for all clips in video
fig, ax = plt.subplots()
ax.hist2d(
    position_flat.sel(space="x").values,
    position_flat.sel(space="y").values,
    bins=[200, 100],
    cmap="viridis",
)
ax.invert_yaxis()
ax.set_xlabel("x (pixels)")
ax.set_ylabel("y (pixels)")
ax.set_aspect("equal")
ax.set_title(ds_video2.video_id)

# %%
# plot all trajectories in one video
fig, ax = plt.subplots()
ax.scatter(
    position_flat.sel(space="x").values,
    position_flat.sel(space="y").values,
)
ax.invert_yaxis()
ax.set_xlabel("x (pixels)")
ax.set_ylabel("y (pixels)")
ax.set_aspect("equal")


# %%%%%%%%%%%%%
# Datatree Operations

# Most xarray computation methods also exist
# as methods on datatree objects
dt.mean(dim=["time", "individuals"])
dt.std(dim="time").compute()  # .compute to get actual values

# Datatree with all first clips of all videos
# The arguments passed to the method are used for every node,
# so the values of the arguments you pass might be valid
# for one node and invalid for another
dt.isel(clip_id=0)

# Datatree with the first 100 frames of all clips
dt.sel(time=slice(0, 100))

# Load all first videos per day into one datatree
leaves_pattern = "*01-Right*"
dt_subset_videos = dt.match(leaves_pattern)


# Scale
# Note: this would also scale confidence
scalebar_in_pixels = 100
scalebar_in_mm = 50
dt * (scalebar_in_pixels / scalebar_in_mm)


# Map over datasets
def scale(ds, factor):
    if "position" in ds:
        return ds.assign(position=ds.position * factor)
    return ds


dt.map_over_datasets(scale, 8)


# %%
# To promote a non-dimension coordinate to dimension coordinate

# Make clip_start_frame_0idx the dimension coordinate instead of clip
# ds_reindexed = ds_video.swap_dims({'clip_id': 'clip_start_frame_0idx'})

# Now this works:
# ds_reindexed.sel(clip_start_frame_0idx=79174)

# %%%%%%%%%%%%%
# Other:
# - compute path length per individual?
# - filter by pathlength?
# - plot all distances to starting point (e.g. if starting point in burrow)
#   for all "loop00"
# - what do I do with a datatree...?


