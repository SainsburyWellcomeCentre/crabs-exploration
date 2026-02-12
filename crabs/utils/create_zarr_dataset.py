"""Combine VIA tracks files into a single zarr dataset.

VIA track files per video are combined into a single movement dataset,
and then saved as a group within a zarr dataset.


"""

import argparse
import sys
import warnings
from collections import defaultdict
from collections.abc import Callable
from pathlib import Path

import numpy as np
import pandas as pd
import xarray as xr
import zarr
from movement.io import load_bboxes
from tqdm import tqdm

# Suppress Zarr V3 warnings
warnings.filterwarnings(
    "ignore",
    category=zarr.errors.UnstableSpecificationWarning,
    message=".*does not have a Zarr V3 specification.*",
)
warnings.filterwarnings(
    "ignore",
    category=zarr.errors.ZarrUserWarning,
    message=(
        ".*Consolidated metadata is currently "
        "not part in the Zarr format 3 specification.*"
    ),
)

DEFAULT_CHUNKS = {"time": 1000, "space": -1, "individuals": -1, "clip_id": -1}


def load_ds_chunked(
    via_tracks_file_path: str | Path,
    df_metadata: pd.DataFrame,
    chunks=None,
) -> xr.Dataset:
    """Read VIA tracks and metadata as a chunked `movement` dataset.

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
    row = df_metadata.loc[df_metadata["loop_clip_name"] == clip_filename].iloc[
        0
    ]
    global_clip_start_frame_0idx = row["loop_START_frame_ffmpeg"] - 1
    global_clip_end_frame_0idx = row["loop_END_frame_ffmpeg"] - 1
    global_escape_start_frame_0idx = row["escape_START_frame_0_based_idx"]

    # -------------------------
    # Add escape_state as data variable
    local_escape_start_frame_0idx = (
        global_escape_start_frame_0idx - global_clip_start_frame_0idx
    )
    # float16 (not int/bool) to allow for NaN padding after
    # concatenating along clip_id
    escape_state = np.zeros(ds.time.shape[0], dtype=np.float16)
    escape_state[local_escape_start_frame_0idx:] = 1.0
    ds["escape_state"] = ("time", escape_state)

    # -------------------------
    # Add clip dimension and associated coordinates
    ds = ds.expand_dims("clip_id")
    ds = ds.assign_coords(
        clip_id=np.array([clip_filename_to_clip_id(clip_filename)], dtype=str)
    )

    # Add clip start, end, escape start and type
    # as non-index coordinates along the clip_id dimension
    ds = ds.assign_coords(
        clip_first_frame_0idx=(
            "clip_id",
            [global_clip_start_frame_0idx],
        ),
        clip_last_frame_0idx=(
            "clip_id",
            [global_clip_end_frame_0idx],
        ),
        clip_escape_first_frame_0idx=(
            "clip_id",
            [global_escape_start_frame_0idx],
        ),
        clip_escape_type=("clip_id", [row["escape_type"].lower()]),
    )

    # -------------------------

    # Chunk the dataset for the clip
    # (underlying arrays become dask arrays)
    # This is to ensure that when concatenated they
    # fit in memory
    chunks = chunks or DEFAULT_CHUNKS
    return ds.chunk(chunks)


def get_video_dataset(
    video_id: str, via_track_files: list[str], df_metadata: pd.DataFrame
) -> xr.Dataset:
    """Load, concatenate, and prepare dataset for a single video.

    Parameters
    ----------
    video_id : str
        Video ID (e.g. "04.09.2023-01-Right")
    via_track_files : list[str]
        List of VIA track file paths for this video
    df_metadata : pd.DataFrame
        Dataframe containing metadata for all clips

    Returns
    -------
    ds : xr.Dataset
        movement dataset containing VIA tracks file for a single video, with
        added video attributes (source files, video_id and fps).

    """
    # Get list of chunked datasets per clip
    list_ds = [load_ds_chunked(f, df_metadata) for f in via_track_files]

    # Concatenate all clip ds along clip_id dimension
    ds = xr.concat(
        list_ds,
        dim="clip_id",
        join="outer",
        coords="different",
        compat="equals",
    )

    # Add video-level attributes (fps, source files, video_id)
    ds = add_video_attrs(video_id, via_track_files, df_metadata, ds)

    # return chunked dataset
    return ds.chunk(DEFAULT_CHUNKS)


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
    grouped_by_key : dict
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
        (e.g. "path/to/04.09.2023-01-Right-Loop05_tracks.csv")

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
        (e.g. "path/to/04.09.2023-01-Right-Loop05_tracks.csv")

    Returns
    -------
    clip_filename : str
        Clip filename (e.g. "04.09.2023-01-Right-Loop05.mp4")

    """
    return Path(via_tracks_path).stem.removesuffix("_tracks") + ".mp4"


def clip_filename_to_clip_id(clip_filename: str | Path) -> str:
    """Return clip ID from clip filename.

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


def add_video_attrs(video_id, via_track_files, df_metadata, ds_combined):
    """Add video data to dataset.

    Parameters
    ----------
    video_id : str
        Video ID (e.g. "04.09.2023-01-Right")
    via_track_files : list[str]
        List of VIA track file paths for this video
    df_metadata : pd.DataFrame
        Dataframe containing metadata for all clips
    ds_combined : xr.Dataset
        movement dataset containing VIA tracks file for a single video.

    Returns
    -------
    ds_combined : xr.Dataset
        movement dataset containing VIA tracks file for a single video, with
        added video attributes (source files, video_id and fps).

    """
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
    ds_combined.attrs["source_file"] = via_track_files

    return ds_combined


def main(args):
    """Create zarr dataset from VIA track files.

    VIA track files per video are combined into a single movement dataset,
    and then saved as a group within a zarr dataset.
    """
    # Initialise zarr store
    root = zarr.open_group(args.zarr_store, mode=args.zarr_mode_store)

    # Read metadata dataframe
    df_metadata = pd.read_csv(args.metadata_csv)

    # Group VIA tracks files per video
    map_video_to_filepaths_and_clips = group_files_per_video(
        args.via_tracks_dir,
        args.via_tracks_glob_pattern,
        parse_video_fn=via_tracks_to_video_filename,
    )

    # Concatenate clips from the same video into one dataset
    pbar = tqdm(map_video_to_filepaths_and_clips.items())
    for video_id, clip_files in pbar:
        # Log info
        pbar.set_description(f"Processing {video_id}")

        # Get video dataset
        ds_combined = get_video_dataset(video_id, clip_files, df_metadata)

        # Rechunk to uniform sizes before saving to zarr
        # (chunk boundaries after concatenating are
        # defined by the length of each "clip_id", and so
        # they are non-uniform. We need to rechunk here)
        ds_combined = ds_combined.chunk({**DEFAULT_CHUNKS, "clip_id": 1})

        # Save group to zarr
        ds_combined.attrs["data_vars_order"] = list(ds_combined.data_vars)
        ds_combined.to_zarr(
            store=root.store,
            group=f"{video_id}",
            mode=args.zarr_mode_group,
        )


def parse_args(args: list[str]) -> argparse.Namespace:
    """Parse command line arguments.

    Parameters
    ----------
    args : list[str]
        Command line arguments.

    Returns
    -------
    argparse.Namespace
        Arguments parsed from command line.

    """
    # Define parser
    parser = argparse.ArgumentParser(
        description="Combine VIA track files into a single zarr dataset"
    )
    parser.add_argument(
        "--via_tracks_dir",
        type=str,
        required=True,
        help="Path to the directory with VIA track files",
    )
    parser.add_argument(
        "--metadata_csv",
        type=str,
        required=True,
        help="Path to the metadata CSV file",
    )
    parser.add_argument(
        "--zarr_store",
        type=str,
        required=True,
        help="Path to the zarr store to create",
    )
    parser.add_argument(
        "--zarr_mode_store",
        type=str,
        default="w-",
        help=(
            "Mode to open zarr store with. "
            "Default: 'w-' (will fail if store exists)."
            "Use 'w' to overwrite existing store "
            "and 'a' to append to existing store."
        ),
    )
    parser.add_argument(
        "--zarr_mode_group",
        type=str,
        default="w-",
        help=(
            "Mode to write to zarr group. "
            "Default: 'w-' (will fail if group exists)."
            "Use 'w' to overwrite existing group "
            "and 'a' to append to existing group."
        ),
    )
    parser.add_argument(
        "--via_tracks_glob_pattern",
        type=str,
        default="*.csv",
        help="Glob pattern to match VIA track files in the directory",
    )

    # Assign list of arguments to parser
    return parser.parse_args(args)


def app_wrapper():
    """Wrap function for extracting loop clips."""
    args = parse_args(sys.argv[1:])
    main(args)


if __name__ == "__main__":
    app_wrapper()
