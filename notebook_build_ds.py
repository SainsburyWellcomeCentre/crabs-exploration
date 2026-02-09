# %%
from collections import defaultdict
from collections.abc import Callable
from pathlib import Path

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


def group_files_per_video(
    files_dir: str | Path,
    glob_pattern: str,
    parse_video_fn: Callable,
):
    """Group filepaths per video.

    Parameters
    ----------
    files_dir : str | Path
        Path to directory containing files
    glob_pattern : str
        Glob pattern to match files in directory
    parse_video_fn: Callable
        Function to parse video name from file path

    Returns
    -------
        grouped_by_key: dict
            Dictionary with video name as key and list of filepaths as value.
            We convert the defaultdict to a dict to prevent key typos creating
            a new key and empty list.

    """
    files = sorted(Path(files_dir).glob(glob_pattern))
    grouped_by_key = defaultdict(list)
    for f in files:
        video_name = parse_video_fn(f)
        grouped_by_key[video_name].append(str(f))
        # make serializable to save later
    return dict(grouped_by_key)


def via_tracks_to_video_filename(via_tracks_path: str | Path) -> str:
    """Return video filename without extension from VIA tracks file path."""
    return Path(via_tracks_path).stem.split("-Loop")[0]


def via_tracks_to_clip_filename(via_tracks_path: str | Path) -> str:
    """Return clip filename with extension from VIA tracks filepath.

    Parameters
    ----------
    via_tracks_path : str | Path
        Path to VIA tracks file

    Returns
    -------
    clip_filename : str
        Clip filename (e.g. "04.09.2023-01-Right-Loop05.mp4")

    """
    return Path(via_tracks_path).stem.removesuffix("_tracks") + ".mp4"


def clip_filename_to_clip_id(clip_filename: str | Path) -> str:
    """Return clip ID (Loop09) from clip filename."""
    return Path(str(clip_filename).rsplit("-")[-1]).stem


def load_ds_and_add_metadata(
    via_tracks_file_path: str | Path, df_metadata: pd.DataFrame
) -> xr.Dataset:
    """Read VIA tracks and metadata as `movement` dataset.

    Args:
        via_tracks_file_path: Path to VIA tracks file
        df_metadata: DataFrame with metadata

    Returns:
        ds: movement dataset with metadata

    """
    # Load VIA tracks file as movement dataset
    ds = load_bboxes.from_via_tracks_file(via_tracks_file_path)

    # Extract metadata for this row
    clip_filename = via_tracks_to_clip_filename(ds.attrs["source_file"])
    row = df_metadata.loc[
        df_metadata["loop_clip_name"] == clip_filename
    ].squeeze()
    global_clip_start_frame_0idx = row["loop_START_frame_ffmpeg"] - 1
    global_clip_end_frame_0idx = row["loop_END_frame_ffmpeg"] - 1
    global_escape_start_frame_0idx = row["escape_START_frame_0_based_idx"]

    # Add clip dimension
    ds = ds.expand_dims({"clip_id": [clip_filename_to_clip_id(clip_filename)]})

    # Add clip start and end as dimensionless coordinates
    # (as non-dim coordinates because they are categorical metadata
    # of each clip, not a measure quantity; after concatenating
    # along clip_id they become non-index coordinates, so they wont
    # interfere with alignment)
    ds = ds.assign_coords(
        clip_start_frame_0idx=global_clip_start_frame_0idx,
        clip_end_frame_0idx=global_clip_end_frame_0idx,
        clip_escape_start_frame_0idx=global_escape_start_frame_0idx,
        clip_escape_type=row["escape_type"].lower(),
    )

    # Add escape state array along time dimension
    local_escape_start_frame_0idx = (
        global_escape_start_frame_0idx - global_clip_start_frame_0idx
    )
    # float16 (not int/bool) to allow for NaN padding after
    # concatenating along clip_id
    escape_state = np.zeros(ds.time.shape[0], dtype=np.float16)
    escape_state[local_escape_start_frame_0idx:] = 1.0
    ds["escape_state"] = ("time", escape_state)

    return ds


# %%
# Build concatenated datasets per video

df_metadata = pd.read_csv(csv_metadata_path)
map_video_to_filepaths_and_clips = group_files_per_video(
    via_tracks_dir,
    "*.csv",
    parse_video_fn=via_tracks_to_video_filename,
)

# Loop thru videos and files
list_ds_videos = []
for video_id, clip_files in map_video_to_filepaths_and_clips.items():
    # Get list of chunked datasets for each file
    list_ds_chunked = [
        load_ds_and_add_metadata(file, df_metadata) for file in clip_files
    ]

    # Concatenate along "clip_id" dimension
    # (the output will be a chunked / dask dataset,
    # a dataset with dask dataarrays)
    ds_combined = xr.concat(
        list_ds_chunked,
        dim="clip_id",
        join="outer",
    )

    # Get fps for this video from metadata
    # (all clips in the same video should share the same fps)
    video_fps_values = df_metadata.loc[
        df_metadata["video_name"].str.removesuffix(".mov") == video_id, "fps"
    ]
    if video_fps_values.nunique() != 1:
        raise ValueError(
            f"Expected uniform fps for video '{video_id}', "
            f"got {video_fps_values.unique()}"
        )
    ds_combined.attrs["fps"] = video_fps_values.iloc[0]

    # Add all source files as attributes
    # (by default only the first ds attrs is retained
    # in the concat output)
    ds_combined.attrs["video_id"] = video_id
    ds_combined.attrs["source_file"] = clip_files

    # Append to list of datasets
    list_ds_videos.append(ds_combined)


# %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
# Plot centroid one clip
# video_name = "04.09.2023-01-Right"
# Q: is there a faster way to plot rather than looping thru individuals?
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
        c="r",
    )
ax.invert_yaxis()
ax.set_xlabel("x (pixels)")
ax.set_ylabel("y (pixels)")
ax.set_aspect("equal")
ax.set_title(
    f"{ds_loop.clip_id.values.item()} - {ds_loop.escape_type.values.item()}"
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
