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
import shutil

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

# DEFAULT_CHUNKS = {"time": 1000, "space": -1, "individuals": -1, "clip_id": 1}


def load_extended_ds(
    via_tracks_file_path: str | Path,
    df_metadata: pd.DataFrame,
) -> xr.Dataset:
    """Read VIA tracks and metadata as a `movement` dataset.

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
        added metadata as coordinates and attributes.

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
    # float16 (not int/bool) to allow for NaN padding after
    # concatenating along clip_id
    escape_state = np.zeros(ds.time.shape[0], dtype=np.float16)
    escape_state[local_escape_start_frame_0idx:] = 1.0
    ds["escape_state"] = ("time", escape_state)

    # Add clip dimension and associated coordinates
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

    # -------------------------
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
        # str to make it serializable to save later
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
    if video_fps_values.nunique() != 1:
        raise ValueError(
            f"Expected uniform fps for video '{video_id}', "
            f"got {video_fps_values.unique()}"
        )
    return video_fps_values.iloc[0]


def _get_encoding_for_chunks(
    ds: xr.Dataset, chunk_sizes: dict[str, int]
) -> dict:
    """Generate encoding dict for zarr chunking.

    Parameters
    ----------
    ds : xr.Dataset
        Dataset to generate encoding for
    chunk_sizes : dict[str, int]
        Mapping from dimension name to chunk size. Use -1 for full dimension.

    Returns
    -------
    encoding : dict
        Encoding dict suitable for ds.to_zarr(encoding=...)

    """
    encoding = {}
    for var in ds.data_vars:
        var_chunks = []
        for dim in ds[var].dims:
            chunk_size = chunk_sizes.get(dim, -1)
            if chunk_size == -1:
                var_chunks.append(ds.sizes[dim])
            else:
                var_chunks.append(chunk_size)
        encoding[var] = {"chunks": tuple(var_chunks)}
    return encoding


def create_proto_zarr_store(args, temp_store):
    # Initialise zarr store
    root = zarr.open_group(temp_store, mode=args.zarr_mode_store)

    # Read metadata dataframe
    df_metadata = pd.read_csv(args.metadata_csv)

    # Group VIA tracks files per video
    map_video_to_filepaths_and_clips = _group_files_per_video(
        args.via_tracks_dir,
        args.via_tracks_glob_pattern,
        parse_video_fn=_via_tracks_to_video_filename,
    )

    # Loop thru videos
    map_video_to_attrs = {}
    for video_id, clip_files in map_video_to_filepaths_and_clips.items():
        # Loop thru clips
        for f in clip_files:
            # Read VIA tracks file as extended movement dataset
            ds_clip = load_extended_ds(f, df_metadata)

            # Get clip_id from the dataset
            clip_id = ds_clip.clip_id.values[0]

            # Write each clip as a separate subgroup
            encoding = _get_encoding_for_chunks(ds_clip, {"time": 1000})
            ds_clip.to_zarr(
                root.store,
                group=f"{video_id}/{clip_id}",
                mode="w",
                encoding=encoding,
            )

        # Save attrs for this video
        map_video_to_attrs[video_id] = {
            "source_file": clip_files,
            "fps": _get_video_fps(video_id, df_metadata),
        }

    return root.store_path, map_video_to_attrs


def main(args):
    """Create zarr dataset from VIA track files.

    VIA track files per video are combined into a single movement dataset,
    and then saved as a group within a zarr dataset.
    """
    temp_store = f"{args.zarr_store}.temp"
    zarr_store_path, map_video_to_attrs = create_proto_zarr_store(
        args, temp_store
    )

    # ---------------------
    # Create final zarr store

    # Read temporary zarr store
    dt = xr.open_datatree(
        zarr_store_path,
        engine="zarr",
        chunks={},
    )
    # Initialise final store on disk
    final_root = zarr.open_group(args.zarr_store, mode=args.zarr_mode_store)

    # Write one video at a time to store
    # Only one video's dask graph exists at a time
    for video_name in tqdm(dt.children, desc="Processing videos"):
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
        print(type(ds_video["position"].data))  # dask.array or numpy.ndarray?

        # Add video attributes
        ds_video.attrs = {
            **map_video_to_attrs[video_name],
            "video_id": video_name,
        }

        # Rechunk after concatenating
        ds_video = ds_video.chunk(
            {"time": 1000, "space": -1, "individuals": -1, "clip_id": 1}
        )

        # Add this dataset as a leaf in the new DataTree
        # dt_restructured[video_name] = xr.DataTree(ds_video)

        # Save to zarr store
        encoding = _get_encoding_for_chunks(
            ds_video, 
            {"time": 1000, "space": -1, "individuals": -1, "clip_id": 1}
        )
        ds_video.to_zarr(final_root.store, group=video_name, mode="w", encoding=encoding)

    # # Save the restructured DataTree to a new zarr store
    # dt_restructured.to_zarr(
    #     final_root.store,
    #     mode="w-",
    # )

    print("Restructured zarr store saved successfully!")
    # ----------------------

    # Delete temp store
    if Path(temp_store).exists():
        try:
            shutil.rmtree(temp_store)
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
        "--zarr_mode_group",  # ----> remove?
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
