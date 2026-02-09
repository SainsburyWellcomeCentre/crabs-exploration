# %%
from collections import defaultdict
from pathlib import Path

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import xarray as xr
from movement.io import load_bboxes
from movement.plots import plot_centroid_trajectory

# %%
via_tracks_dir = "/Users/sofia/arc/project_Zoo_crabs/loops_tracking_above_10th_percentile_slurm_1825237_SAMPLE"
csv_metadata_path = "/Users/sofia/arc/project_Zoo_crabs/CrabsField/crab-loops/loop-frames-ffmpeg.csv"

# Can I fetch csv data from GIN?

# %%
%matplotlib widget


# %%
# Helper functions

def load_ds_and_add_metadata(via_tracks_file_path, df_metadata):
    """Read movement dataset and add metadata."""
    # Load VIA tracks file as movement dataset
    ds = load_bboxes.from_via_tracks_file(via_tracks_file_path)

    # Get metadata from csv
    clip_name = (
        Path(ds.attrs["source_file"]).stem.removesuffix("_tracks") + ".mp4"
    )

    # Extract metadata for this row
    row = df_metadata[df_metadata["loop_clip_name"] == clip_name]
    global_clip_start_frame_0idx = row["loop_START_frame_ffmpeg"].item() - 1
    global_clip_end_frame_0idx = row["loop_END_frame_ffmpeg"].item() - 1
    global_escape_start_frame_0idx = row[
        "escape_START_frame_0_based_idx"
    ].item()
    escape_type = row["escape_type"].item()
    fps = row["fps"].item()  # cast to float32?

    # Add metadata to ds
    # Add clip dimension
    ds = ds.expand_dims({"clip_id": [Path(clip_name).stem.rsplit("-", 1)[-1]]})

    # Add clip start and end as dimensionless coordinates
    # as non-dim coordinates because they are categorical metadata
    # of each clip, not a measure quantity
    ds = ds.assign_coords(
        {
            "clip_start_frame_0idx": global_clip_start_frame_0idx,
            "clip_end_frame_0idx": global_clip_end_frame_0idx,
            "clip_escape_start_frame_0idx": global_escape_start_frame_0idx,
            "clip_escape_type": escape_type.lower(),
        }
    )

    # Add fps as attributes (TODO: assign per video instead)
    ds.attrs["fps"] = fps

    # Add state array along time dimension
    local_escape_start_frame = (
        global_escape_start_frame_0idx - global_clip_start_frame_0idx
    )
    ds["escape_state"] = (
        "time",
        np.r_[
            np.zeros(local_escape_start_frame - 1, dtype=np.float16),
            np.ones(
                ds.time.shape[0] - (local_escape_start_frame - 1),
                dtype=np.float16,
            ),
        ],
    )

    # Add attributes with unique name?

    return ds


def get_map_video_to_files(via_tracks_dir, via_tracks_glob_pattern):
    """Return dict from video name to files."""
    # Get list of VIA track files
    list_files = sorted(
        list(Path(via_tracks_dir).glob(via_tracks_glob_pattern))
    )

    # Get mapping from video name to files
    # ----------
    # Can i make this more general?
    map_video_to_files = defaultdict(list)
    for f in list_files:
        map_video_to_files[Path(f).stem.split("-Loop")[0]].append(f)
    # ----------
    return map_video_to_files


# %%
# Build concatenated datasets per video

df_metadata = pd.read_csv(csv_metadata_path)
map_video_to_files = get_map_video_to_files(via_tracks_dir, "*.csv")

list_ds_videos = []
# Loop thru videos and files
for video_id, files in map_video_to_files.items():
    # --------
    # Get clip names (e.g "Loop09")
    # Can I make this more general?
    clip_names = [Path(f).stem.rsplit("-")[-1].split("_")[0] for f in files]
    # ----------

    # Get list of chunked datasets for each file
    list_ds_chunked = [load_ds_and_add_metadata(f, df_metadata) for f in files]

    # Concatenate along "loop" dimension
    # (the output will be a chunked / dask dataset,
    # a dataset with dask dataarrays)
    ds_combined = xr.concat(
        list_ds_chunked,
        dim="clip_id",
        join="outer",
        # change how attrs are retained
    )

    # Add attributes
    ds_combined.attrs["source_file"] = [str(f) for f in files]

    list_ds_videos.append(ds_combined)


# %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
# Plot centroid one clip
# video_name = "04.09.2023-01-Right"
ds_loop = list_ds_videos[2].isel(clip_id=0)

# plot all individuals
fig, ax = plt.subplots()
colors = plt.cm.tab20.colors
for i, ind in enumerate(ds_loop.individuals):
    # plot outbound
    plot_centroid_trajectory(
        ds_loop.position.where(ds_loop.escape_state == 0.0, drop=True),
        individual=ind,
        ax=ax,
        c=colors[i % len(colors)],
    )
    # plot inbound
    plot_centroid_trajectory(
        ds_loop.position.where(ds_loop.escape_state == 1.0, drop=True),
        individual=ind,
        ax=ax,
        c='r',
    )
ax.invert_yaxis()
ax.set_xlabel("x (pixels)")
ax.set_ylabel("y (pixels)")
ax.set_aspect("equal")
ax.set_title(f"{ds_loop.clip_id.values.item()} - {ds_loop.escape_type.values.item()}")



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
# plot all paths for one video
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
print(ds_video.where(ds_video.escape_type == "spontaneous", drop=True).clip_id.values)

# Get loops with "triggered" escape
print(ds_video.where(ds_video.escape_type == "triggered", drop=True).clip_id.values)


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
