"""Demo notebook for working with crab dataset."""

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

# Hide attributes globally
xr.set_options(display_expand_attrs=False)


# %%
# Input data

# %%%%%%%%%%%%%
# Read dataset as a datatree

dt = xr.open_datatree(
    "all_trials_per_video_1.zarr",
    engine="zarr",
    chunks={},
)


# %%
# If dataset saved with group being video/clip, to concatenate per video:
# Create a new DataTree where each video becomes a leaf node
# with concatenated clips
dt_restructured = xr.DataTree()

for video_name in dt.children:
    # Get all clips for this video and concatenate them
    dt_video = dt[video_name]

    # Concatenate all clip datasets along the clip_id dimension
    ds_video = xr.concat(
        [clip_node.to_dataset() for clip_node in dt_video.leaves],
        dim="clip_id",
        join="outer",
        coords="different",
        compat="equals",
    )

    # Add video attributes
    # ....

    # Rechunk after concatenating
    ds_video = ds_video.chunk(
        {"time": 1000, "space": -1, "individuals": -1, "clip_id": 1}
    )

    # Add this dataset as a leaf in the new DataTree
    dt_restructured[video_name] = xr.DataTree(ds_video)

# Save the restructured DataTree to a new zarr store
dt_restructured.to_zarr(
    "all_trials_per_video_restructured.zarr",
    mode="w-",
)

print("Restructured zarr store saved successfully!")


# %%%%%%%%%%%%%
# # Compare memory before and after loading dask array
# process = psutil.Process(os.getpid())

# # Check memory before
# mem_before = process.memory_info().rss / 1_000_000_000
# print(f"Memory before: {mem_before:.2f} GB")

# # Load the data
# ds_loaded = ds.load()

# # Check memory after
# mem_after = process.memory_info().rss / 1_000_000_000
# print(f"Memory after: {mem_after:.2f} GB")
# print(f"Memory increase: {(mem_after - mem_before):.2f} GB")


# %%
dt = xr.open_datatree(
    "all_trials_per_video_restructured.zarr",
    engine="zarr",
    chunks={},
)


# %%%%%%%%%%%%%%
# Inspect structure
print(list(dt.children.keys()))  # ---> prints lists of videos
print(f"Depth: {dt.depth}")  # 1

# %%
print("Flat list of all paths:")
print(*dt.groups, sep="\n")

print("List of leaf paths only:")
print(*[node.path for node in dt.leaves], sep="\n")

print("Inspect dimension of leaves:")
for node in dt.leaves:
    if node.has_data:
        print(
            f"{node.path}: dims={dict(node.sizes)}, "
            f"vars={list(node.data_vars)}"
        )


# %%
# With non-indexed coords:

# To load "clip_id" coordinates
# .compute returns a new object, .load modifies in place
for node in dt.leaves:
    node.coords["clip_id"].load()


# %%
# how to get all "triggered" from one video?
ds = dt["04.09.2023-01-Right"].to_dataset()
ds.where(ds.clip_escape_type == "triggered", drop=True)

# To order data variables
ds = dt["04.09.2023-01-Right"].to_dataset()
ds = ds[ds.attrs["data_vars_order"]]

# to show stats of confidence values per clip
ds.confidence.mean().compute()
ds.confidence.std().compute()
ds.confidence.min().compute()
ds.confidence.max().compute()
ds.confidence.median(dim=("time", "individuals")).compute()


# %%

dt["04.09.2023-01-Right"].coords["clip_id"]

# %%


# %%%%%%%%%%%%%
# Get escape type for loop at index 1
print(ds_combined.isel(loop=1).escape_type.values.item())

# Get loops with "spontaneous" escape
print(
    ds_combined.where(
        ds_combined.escape_type.compute() == "spontaneous", drop=True
    )
)

# Get loops with "triggered" escape
print(
    ds_combined.where(
        ds_combined.escape_type.compute() == "triggered", drop=True
    )
)

# %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
# Plot centroid one clip
# video_name = "04.09.2023-01-Right"

# Select a clip
ds_clip = list_ds_videos[0].isel(clip_id=0)

# Pre-compute individual -> color mapping
colors = plt.cm.tab20.colors
color_map = {
    ind.item(): colors[i % len(colors)]
    for i, ind in enumerate(ds_clip.individuals)
}

# Outbound data (escape_state == 0)
pos_out = ds_clip.position.where(ds_clip.escape_state == 0.0, drop=True)
pos_out_flat = pos_out.stack(flat=("time", "individuals")).dropna("flat")

# color assignment per individual
ind_labels = pos_out_flat.individuals.values
c_out = np.array([color_map[ind] for ind in ind_labels])

# Inbound data (escape_state == 1)
pos_in = ds_clip.position.where(ds_clip.escape_state == 1.0, drop=True)
pos_in_flat = pos_in.stack(flat=("time", "individuals")).dropna("flat")

# color assignment per individual
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
# plot all individuals
fig, ax = plt.subplots()
colors = plt.cm.tab20.colors
for i, ind in enumerate(ds_clip.individuals):
    # plot outbound
    plot_centroid_trajectory(
        ds_clip.position.where(ds_clip.escape_state == 0.0, drop=True),
        individual=ind,
        ax=ax,
        c=colors[i % len(colors)],
    )
    # plot inbound
    plot_centroid_trajectory(
        ds_clip.position.where(ds_clip.escape_state == 1.0, drop=True),
        individual=ind,
        ax=ax,
        c="r",
    )
ax.invert_yaxis()
ax.set_xlabel("x (pixels)")
ax.set_ylabel("y (pixels)")
ax.set_aspect("equal")
ax.set_title(
    f"{ds_clip.clip_id.values.item()} - {ds_clip.clip_escape_type.values.item()}"
)


# %%%%%%%%%
# Plot occupancy for all clips in same video

# select video
ds_video = list_ds_videos[2]

# prepare data
# flatten to just time and space coords
position_flat = ds_video.position.stack(
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
# ax.set_title(video_name)

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
# Get escape type for loop at index 1
print(ds_video.isel(clip_id=1).escape_type.values.item())

# Get loops with "spontaneous" escape
print(
    ds_video.where(
        ds_video.escape_type == "spontaneous", drop=True
    ).clip_id.values
)

# Get loops with "triggered" escape
print(
    ds_video.where(
        ds_video.escape_type == "triggered", drop=True
    ).clip_id.values
)


# %%
# Can you promote a non-dimension coordinate to dimension coordinate?

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


# %%%%%%%%%%%%%
# Operations

# Most xarray computation methods also exist
# as methods on datatree objects
dt.mean(dim=["time", "individuals"])
dt.std(dim="time").compute()  # .compute to get actual values

# Datatree with all first loops
# The arguments passed to the method are used for every node,
# so the values of the arguments you pass might be valid
# for one node and invalid for another
dt.isel(loop=0)

# Datatree with the first 100 frames of all loops / clips
dt.sel(time=slice(0, 100))

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


# Load all first videos per day into one datatree
leaves_pattern = "*01-Right*"
dt_subset_videos = dt.match(leaves_pattern)


# %%
