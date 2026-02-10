# %%
from collections import defaultdict
from collections.abc import Callable
from pathlib import Path

import numpy as np
import pandas as pd
import xarray as xr
import zarr
from movement.io import load_bboxes

# %%%%%%%%%
# Input data
via_tracks_dir = "/Users/sofia/arc/project_Zoo_crabs/loops_tracking_above_10th_percentile_slurm_1825237_SAMPLE"
csv_metadata_path = "/Users/sofia/arc/project_Zoo_crabs/CrabsField/crab-loops/loop-frames-ffmpeg.csv"

# Can I fetch csv data from GIN?

zarr_store_path = "all_trials_per_video.zarr"


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
        # str to make it serializable to save later
    return dict(grouped_by_key)


def via_tracks_to_video_filename(via_tracks_path: str | Path) -> str:
    """Return video filename without extension from VIA tracks file path.

    Parameters
    ----------
    via_tracks_path : str | Path
        Path to VIA tracks file

    Returns
    -------
    video_filename : str
        Video filename without extension (e.g. "04.09.2023-01-Right")
    """
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
    """Return clip ID (Loop09) from clip filename.

    Parameters
    ----------
    clip_filename : str | Path
        Clip filename (e.g. "04.09.2023-01-Right-Loop05.mp4")

    Returns
    -------
    clip_id : str
        Clip ID (e.g. "Loop05")

    """
    return Path(str(clip_filename).rsplit("-")[-1]).stem


def load_ds_add_metadata_chunk(
    via_tracks_file_path: str | Path, df_metadata: pd.DataFrame, chunks=None
) -> xr.Dataset:
    """Read VIA tracks and metadata as a `movement` dataset.

    Parameters
    ----------
    via_tracks_file_path : str | Path
        Path to VIA tracks file
    df_metadata : pd.DataFrame
        Dataframe containing metadata for all clips
    chunks : dict, optional
        Dictionary specifying chunk sizes for each dimension, by default None
         (if None, default chunk size of 1000 along time dimension is used)

    Returns
    -------
    ds : xr.Dataset
        Dataset containing movement data from VIA tracks file, with
        added metadata as coordinates and attributes. The dataset is
        chunked according to the specified chunk sizes.

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

    # Add clip start, end, escape start and type as dimensionless coordinates
    # (because they are categorical metadata
    # of each clip, not a measure quantity; after concatenating
    # along clip_id they become non-index coordinates mirroring clip_id,
    # so they wont interfere with alignment)
    ds = ds.assign_coords(
        clip_start_frame_0idx=global_clip_start_frame_0idx,
        # clip_end_frame_0idx=global_clip_end_frame_0idx,
        # clip_escape_start_frame_0idx=global_escape_start_frame_0idx,
        # clip_escape_type=row["escape_type"].lower(),
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

    # Chunk the dataset
    # (underlying arrays become dask arrays)
    # TODO: Ensure scalar coordinates are not chunked
    if chunks is None:
        chunks = {"time": 1000, "individuals": -1, "space": -1}

    return ds.chunk(chunks)


# %%%%%%%%%%%%%%%%%%%%%%%%
# Build dataset

# Initialise zarr store
root = zarr.open_group(zarr_store_path, mode="w")


# Read metadata dataframe
df_metadata = pd.read_csv(csv_metadata_path)

# Group VIA tracks files per video
map_video_to_filepaths_and_clips = group_files_per_video(
    via_tracks_dir,
    "*.csv",
    parse_video_fn=via_tracks_to_video_filename,
)


# Concatenate clips from the same video into one dataset
list_ds_videos = []
for video_id, clip_files in map_video_to_filepaths_and_clips.items():
    # Get list of chunked datasets for each file
    list_ds_chunked = [
        load_ds_add_metadata_chunk(file, df_metadata) for file in clip_files
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
    # list_ds_videos.append(ds_combined)

    # Rechunk to uniform sizes before saving to zarr
    # (chunk boundaries after concatenating are
    # defined by the length of each "loop", and so
    # they are non-uniform. We need to rechunk here)
    ds_combined = ds_combined.chunk(
        {
            "clip_id": 1,
            "time": 1000,
            "space": -1,
            "individuals": -1,
            # "clip_start_frame_0idx": 1,
            # "clip_end_frame_0idx": 1,
            # "clip_escape_start_frame_0idx": 1,
            # "clip_escape_type": 1,
        }
        # -1 meaning all dimensions are included in a chunk
    )

    # # Use encoding to control how non-index coordinates
    # # are chunked
    # non_index_coords = [
    #     c for c in ds_combined.coords if c not in ds_combined.indexes
    # ]
    # for coord in non_index_coords:
    #     ds_combined[coord] = ds_combined[coord].load()

    # Save to zarr under video group
    # encoding = {
    #     "position": {"chunks": (1, 1000, -1, -1)},
    #     "shape": {"chunks": (1, 1000, -1, -1)},
    #     "confidence": {"chunks": (1, 1000, -1)},
    #     "escape_state": {"chunks": (1, 1000)},
    #     "clip_start_frame_0idx": {"chunks": (1)},
    #     "clip_end_frame_0idx": {"chunks": (1)},
    #     "clip_escape_start_frame_0idx": {"chunks": (1)},
    #     "clip_escape_type": {"chunks": (1)},
    # }
    # ds_combined.chunk(
    #     chunks={
    #         "clip_id": 1,
    #         "time": 1000,
    #         "space": -1,
    #         "individuals": -1,
    #     },
    # )
    # non_index_coord_chunked = ds_combined.from_array(
    #     ds_combined["clip_start_frame_0idx"].values, chunks=(1)
    # )
    # ds_combined = ds_combined.assign_coords(
    #     clip_start_frame_0idx=(["clip_id"], non_index_coord_chunked)
    # )

    ds_combined.to_zarr(root.store, group=f"{video_id}", consolidated=True)

# %%
