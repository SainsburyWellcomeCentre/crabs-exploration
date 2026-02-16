"""Combine VIA tracks files into a single zarr dataset.

We first create a temporary zarr store with each group holding the movement
dataset for a single clip. Then we read from that store and restructure it so
that each group holds the movement dataset for a single video, which is the
concatenation of all the clip datasets within that video.
"""

import argparse
import shutil
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


DEFAULT_CHUNK_SIZES = {
    "time": 1000,
    "space": -1,
    "individuals": -1,
    "clip_id": 1,
}


def load_extended_ds(
    via_tracks_file_path: str | Path,
    df_metadata: pd.DataFrame,
) -> xr.Dataset:
    """Combine VIA tracks file and metadata as a `movement` dataset.

    Parameters
    ----------
    via_tracks_file_path : str | Path
        Path to VIA tracks file
    df_metadata : pd.DataFrame
        Dataframe containing metadata for all clips

    Returns
    -------
    ds : xr.Dataset
        Dataset containing movement data from VIA tracks file, with
        added metadata as coordinates and data variables.

    """
    # Load VIA tracks file as movement dataset
    ds = load_bboxes.from_via_tracks_file(via_tracks_file_path)

    # Extract metadata for this row
    clip_filename = _via_tracks_to_clip_filename(ds.attrs["source_file"])
    row = df_metadata.loc[df_metadata["loop_clip_name"] == clip_filename].iloc[
        0
    ]
    global_clip_start_frame_0idx = row["loop_START_frame_ffmpeg"] - 1
    global_clip_end_frame_0idx = row["loop_END_frame_ffmpeg"] - 1
    global_escape_start_frame_0idx = row["escape_START_frame_0_based_idx"]

    # Add escape_state as data variable
    local_escape_start_frame_0idx = (
        global_escape_start_frame_0idx - global_clip_start_frame_0idx
    )
    # we use float16 (not int/bool) to allow for NaN padding after
    # concatenating along clip_id
    escape_state = np.zeros(ds.time.shape[0], dtype=np.float16)
    escape_state[local_escape_start_frame_0idx:] = 1.0
    ds["escape_state"] = ("time", escape_state)

    # Add clip_id dimension and associated coordinates
    ds = ds.expand_dims("clip_id")
    ds = ds.assign_coords(
        clip_id=np.array([_clip_filename_to_clip_id(clip_filename)], dtype=str)
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
        clip_escape_type=(
            "clip_id",
            np.array([row["escape_type"].lower()], dtype="<U11"),
        ),
    )

    return ds


def _group_files_per_video(
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
        # str to make it serializable when saving later
    return dict(grouped_by_key)


def _via_tracks_to_video_filename(via_tracks_path: str | Path) -> str:
    """Return video filename without extension from VIA tracks file path.

    "path/to/04.09.2023-01-Right-Loop05_tracks.csv" --> "04.09.2023-01-Right"
    """
    return Path(via_tracks_path).stem.split("-Loop")[0]


def _via_tracks_to_clip_filename(via_tracks_path: str | Path) -> str:
    """Return clip filename with extension from VIA tracks filepath.

    "path/to/04.09.2023-01-Right-Loop05_tracks.csv" -->
    "04.09.2023-01-Right-Loop05.mp4"
    """
    return Path(via_tracks_path).stem.removesuffix("_tracks") + ".mp4"


def _clip_filename_to_clip_id(clip_filename: str | Path) -> str:
    """Return clip ID from clip filename.

    "04.09.2023-01-Right-Loop05.mp4" --> "Loop05"
    """
    return Path(str(clip_filename).rsplit("-")[-1]).stem


def _get_video_fps(video_id: str, df_metadata: pd.DataFrame) -> float:
    """Extract video fps from metadata dataframe."""
    video_fps_values = df_metadata.loc[
        df_metadata["video_name"].str.removesuffix(".mov") == video_id, "fps"
    ]
    # all clips in the df for this video should have same fps
    if video_fps_values.nunique() != 1:
        raise ValueError(
            f"Expected uniform fps for video '{video_id}', "
            f"got {video_fps_values.unique()}"
        )
    return video_fps_values.iloc[0]


def create_temp_zarr_store(
    temp_zarr_store: str,
    temp_zarr_mode_store: str,
    temp_zarr_mode_group: str,
    via_tracks_dir: str | Path,
    via_tracks_glob_pattern: str,
    metadata_csv: str | Path,
) -> tuple[Path, dict]:
    """Create a temporary zarr store.

    We define a temporary zarr store with groups = f"{video_id}/{clip_id}"
    (i.e., each group holding the `movement` dataset for a single clip).
    This way, we only have ~1 clip in memory as a `movement` dataset at a
    time.

    We later restructure this temporary store as one with
    groups = f"{video_id}", with each group holding the `movement`
    dataset for a single video. This flatter structure is preferred for easier
    user interaction with the dataset. We do the zarr store creation in two
    steps to avoid out-of-memory issues that would occur when concatenating
    all clip datasets per video if all clips were in memory at the same time.

    """
    # Initialise zarr store
    root = zarr.open_group(temp_zarr_store, mode=temp_zarr_mode_store)

    # Read metadata dataframe
    df_metadata = pd.read_csv(metadata_csv)

    # Group VIA tracks files per video
    map_video_to_filepaths_and_clips = _group_files_per_video(
        via_tracks_dir,
        via_tracks_glob_pattern,
        parse_video_fn=_via_tracks_to_video_filename,
    )

    # Loop thru videos
    map_video_to_attrs = {}
    pbar_videos = tqdm(map_video_to_filepaths_and_clips.items())
    for video_id, clip_files in pbar_videos:
        pbar_videos.set_description(
            f"Temporary processing of clips in video {video_id}"
        )

        # Loop thru clips,
        # add each video_id/clip_id as a group
        for f in clip_files:
            # Read VIA tracks file as extended movement dataset
            ds_clip = load_extended_ds(f, df_metadata)

            # Get clip_id from the dataset
            clip_id = ds_clip.clip_id.values[0]

            # Write each clip as a separate subgroup
            ds_clip = ds_clip.chunk(DEFAULT_CHUNK_SIZES)
            ds_clip.to_zarr(
                root.store,
                group=f"{video_id}/{clip_id}",
                mode=temp_zarr_mode_group,
                # encoding=encoding,
            )

        # Save attrs for this video
        map_video_to_attrs[video_id] = {
            "source_file": clip_files,
            "fps": _get_video_fps(video_id, df_metadata),
        }

    # Get path to temp store
    temp_path = Path(str(root.store_path).replace("file://", ""))

    return temp_path, map_video_to_attrs


def create_final_zarr_store(
    temp_zarr_store: str | Path,
    map_video_to_attrs: dict,
    zarr_store: str | Path,
    zarr_mode_store: str,
    zarr_mode_group: str,
):
    """Create final zarr store from a temporary one.

    In the final zarr store, each group holds the `movement` dataset for a
    single video, which is the concatenation of all the clip datasets
    for that video. We also add the video-level attributes to the video
    dataset.

    """
    # Read temporary zarr store
    dt = xr.open_datatree(
        temp_zarr_store,
        engine="zarr",
        chunks={},
    )
    # Initialise final store on disk
    final_root = zarr.open_group(zarr_store, mode=zarr_mode_store)

    # Write one video at a time to final store
    pbar = tqdm(dt.children)
    for video_name in pbar:
        pbar.set_description(f"Final processing of video {video_name}")

        # Get sub-datatree with all clips for this video
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
        ds_video.attrs = {
            **map_video_to_attrs[video_name],
            "video_id": video_name,
        }

        # Rechunk the dask dataset currently in memory
        # to align dask chunks with desired Zarr chunks
        ds_video = ds_video.chunk(DEFAULT_CHUNK_SIZES)

        # Save to zarr store
        # (xarray will automatically use the dask chunk sizes as the
        # zarr chunk sizes when writing to disk.)
        ds_video.to_zarr(
            final_root.store,
            group=video_name,
            mode=zarr_mode_group,
        )


def main(args):
    """Create zarr dataset from VIA track files.

    In the final dataset, each video is a group. To do this without
    OOM issues, we first create a temporary zarr store with each group
    being a clip, then read from that store and restructure it.
    """
    # Create temporary zarr store, with each group
    # holding the `movement` dataset for a single clip
    temp_zarr_store, map_video_to_attrs = create_temp_zarr_store(
        f"{args.zarr_store}.temp",
        temp_zarr_mode_store=args.zarr_mode_store,
        temp_zarr_mode_group=args.zarr_mode_group,
        via_tracks_dir=args.via_tracks_dir,
        via_tracks_glob_pattern=args.via_tracks_glob_pattern,
        metadata_csv=args.metadata_csv,
    )

    # Create final zarr store
    create_final_zarr_store(
        temp_zarr_store,
        map_video_to_attrs,
        zarr_store=args.zarr_store,
        zarr_mode_store=args.zarr_mode_store,
        zarr_mode_group=args.zarr_mode_group,
    )

    # Delete temp store
    if Path(temp_zarr_store).exists():
        try:
            shutil.rmtree(temp_zarr_store)
        except Exception as e:
            print(f"Warning: Failed to delete temp store: {e}")


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
        help=(
            "Path to the zarr store to create. "
            "The final zarr store will be created at this path, "
            "and a temporary zarr store will be created at <zarr_store>.temp "
            "during processing and deleted at the end."
        ),
    )
    parser.add_argument(
        "--zarr_mode_store",
        type=str,
        default="w-",
        help=(
            "Mode to open zarr store with. "
            "It applies to both the temporary and final zarr store. "
            "Default: 'w-' (will fail if store exists)."
            "Use 'w' to overwrite existing store "
            "and 'a' to append to existing store."
            "If running an array job in the cluster, where each job creates "
            "a zarr group for a single video, use 'a' to append to the same "
            "store across jobs."
        ),
    )
    parser.add_argument(
        "--zarr_mode_group",
        type=str,
        default="w-",
        help=(
            "Mode to write to zarr group. "
            "It applies to both the temporary and final zarr group. "
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
