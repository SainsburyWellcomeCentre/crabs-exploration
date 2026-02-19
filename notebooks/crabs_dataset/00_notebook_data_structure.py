"""Demo notebook for working with crab dataset.

Useful references:
- https://docs.xarray.dev/en/latest/api/datatree.html

"""

# %%
from pathlib import Path

import numpy as np
import xarray as xr

# Hide attributes globally
xr.set_options(
    display_expand_attrs=False,
    display_style="html",  # "text" for readibility in dark mode?
)


# %%
# %matplotlib widget
# %matplotlib osx

# %%%%%%%%%%%%%%%%
# Input data
data_dir = Path("/Users/sofia/swc/CrabTracks")
crabs_zarr_dataset = (
    data_dir
    / "CrabTracks-slurm2412462-slurm2423692.zarr"  # "CrabTracks-slurm2370554-slurm2382788.zarr"
)

data_vars_order = [
    "position",
    "shape",
    "confidence",
    "escape_state",
]

# %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
# Read dataset as an xarray datatree

dt = xr.open_datatree(
    crabs_zarr_dataset,
    engine="zarr",
    chunks={},
)

print(dt)

# %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
# Inspect datatree
# - each group is a video dataset

print(f"Depth: {dt.depth}")

# Number of leaves (i.e. videos)
# ATT: dt.groups includes the root group
print(f"Number of groups: {len(dt.leaves)}")

# Print clips per video
for ds_video in dt.leaves:
    print(f"{ds_video.path}: {len(ds_video.clip_id)} clips")


# %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
# Inspect a single video
# - convert to dataset
# - dimensions, coords and data variables

# .to_dataset(): makes a copy
# .ds(): returns a view, read-only
ds_video = dt["04.09.2023-01-Right"].to_dataset()
ds_video = ds_video[data_vars_order]  # reorder data vars

# Dimensions, coordinates and data vars
# Untracked coordinates
# shows dask arrays unloaded
print(ds_video)

# %%
dt["04.09.2023-01-Right"]  # --> returns a datatree
dt["04.09.2023-01-Right"].ds  # --> returns a dataset view

# %%%%%%%%%%%%%%%%%%%%%%%%%%
# Dask arrays inside datasets

# - Load coordinates only
#  (any coord will do, we choose clip_escape_first_frame_0idx)
# - Check clip_id metadta

# .load(): applies in-place,
# .compute(): returns a new object
ds_video.coords["clip_escape_first_frame_0idx"].load()
print(ds_video)


# %%%%%%%%%%%%%%%%%%%%%%%%%
# Check video dataset attributes

print("Dataset attributes:")
print(*ds_video.attrs.items(), sep="\n")


# %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
# Inspect clips in a video

# Number of clips in this video
print("Number of clips:", ds_video.coords["clip_id"].size)

# Types of escape
print("Escape types:", np.unique(ds_video.clip_escape_type.values))

# Number of triggered and spontaneous clips
ds_video_triggered = ds_video.where(
    ds_video.clip_escape_type == "triggered", drop=True
)
ds_video_spontaneous = ds_video.where(
    ds_video.clip_escape_type == "spontaneous", drop=True
)

print("Number of triggered clips:", ds_video_triggered.coords["clip_id"].size)
print(
    "Number of spontaneous clips:", ds_video_spontaneous.coords["clip_id"].size
)

# %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
# Examples of selection in a video dataset
# - demo sel and isel

# Get escape type for 'Loop09'
print(ds_video.sel(clip_id="Loop09").clip_escape_type.values.item())


# Get escape type for loop at index 1
print(ds_video.isel(clip_id=1).clip_escape_type.values.item())

# %%%%%%%%%%%%%%%%%%%%%%%%%%%
# Simple stats per video

# Confidence per clip in video
ds_video.confidence.median(dim=("time", "individuals"), skipna=True).compute()


# %%%%%%%%%%%%%%%%%%%%%%
# Simple stats per clip
ds_clip = ds_video.isel(clip_id=0)
# or equivalently: ds_clip = ds_video2.sel(clip_id="Loop00")


# to show stats of confidence values per clip
ds_clip.confidence.mean().compute()
ds_clip.confidence.std().compute()
ds_clip.confidence.min().compute()
ds_clip.confidence.max().compute()
ds_clip.confidence.median(dim=("individuals")).compute()


# %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
# Examples of selection in datatree

# Select all videos from one day
leaves_pattern = "07.09.2023*"
dt_subset_videos = dt.match(leaves_pattern)
dt_subset_videos

# %%
# Select all first videos per day into one datatree
leaves_pattern = "*01-Right*"
dt_subset_first_videos = dt.match(leaves_pattern)
dt_subset_first_videos


# %%
# Select only the first clip per video
# Note: The arguments passed to the method are used for every node,
# so the values of the arguments you pass might be valid
# for one node and invalid for another
dt_subset_first_clip = dt.isel(clip_id=0)
dt_subset_first_clip

# %%
# Select only the first 100 frames of all clips
dt_subset_100frames = dt.sel(time=slice(0, 100))
dt_subset_100frames

# Then we can use the subset datatree as e.g.:
# > for node in dt_subset_videos.leaves:
# >    position = node.ds.position

# %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
# Apply functions to all datasets in tree

# Apply scaling to the whole dataset
# CAUTION: confidence is also scaled!
scalebar_in_pixels = 100
scalebar_in_mm = 50
dt * (scalebar_in_pixels / scalebar_in_mm)


# Apply a custom function of all datasets in tree
def scale(ds, factor=1.0):
    if "position" in ds:
        return ds.assign(position=ds.position * factor)
    return ds


dt_scaled = dt.map_over_datasets(scale, kwargs={"factor": 3})


# %%
# Most xarray computation methods also exist
# as methods on datatree objects

# Compute mean "confidence" per clip
dt_mean = dt.mean(dim=["time", "individuals"], skipna=True)
# for one video:
dt_mean["04.09.2023-01-Right"].confidence.compute()


# Compute std
dt_std = dt.std(dim="time", skipna=True)
# for position on one video, first clip (dims: space, individuals)
dt_std["06.09.2023-01-Right"].position.isel(
    clip_id=0
).compute()  # .compute to get actual values


# %%
# To promote a non-dimension coordinate to dimension coordinate

# Make clip_start_frame_0idx the dimension coordinate instead of clip
# ds_reindexed = ds_video.swap_dims({'clip_id': 'clip_start_frame_0idx'})

# Now this works:
# ds_reindexed.sel(clip_start_frame_0idx=79174)
