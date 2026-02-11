from pathlib import Path

import dask
import pandas as pd
import pytest

from crabs.utils.create_zarr_dataset import (
    clip_filename_to_clip_id,
    get_video_dataset,
    group_files_per_video,
    load_ds_chunked,
    via_tracks_to_clip_filename,
    via_tracks_to_video_filename,
)


@pytest.fixture
def sample_via_tracks_file_factory(tmp_path):
    """Return factory of VIA tracks files with custom filenames."""

    def _sample_via_tracks_file(filename):
        """Create a mock VIA tracks file."""
        # Add a few frames with bounding box annotations
        via_data = []
        for frame_idx in [0, 10, 20, 50, 60, 80, 100]:
            via_data.append(
                {
                    "filename": f"frame_{frame_idx:08d}.png",
                    "file_size": 26542080,
                    "file_attributes": '{"foo":5}',
                    "region_count": 1,
                    "region_id": 0,
                    "region_shape_attributes": (
                        '{"name":"rect","x":100.0,"y":100.0,"width":50,"height":50}'
                    ),
                    "region_attributes": '{"track":"1"}',
                }
            )
        df = pd.DataFrame(via_data)

        # Save to CSV file with the expected naming pattern
        csv_path = tmp_path / f"{filename}"
        df.to_csv(csv_path, index=False)
        return Path(csv_path)

    return _sample_via_tracks_file


@pytest.fixture
def sample_metadata_df_factory():
    """Create a mock metadata dataframe.

    It should include the mock VIA tracks file.
    """

    def _sample_metadata_df(list_clip_names):
        n_clips = len(list_clip_names)
        return pd.DataFrame(
            {
                "loop_clip_name": list_clip_names,
                "loop_START_frame_ffmpeg": [100] * n_clips,  # 1-indexed
                "loop_END_frame_ffmpeg": [200] * n_clips,  # 1-indexed
                "escape_START_frame_0_based_idx": [150] * n_clips,  # 0-indexed
                "escape_type": ["spontaneous"] * n_clips,
                "video_name": ["04.09.2023-01-Right.mov"] * n_clips,
                "fps": [30.0] * n_clips,
            }
        )

    return _sample_metadata_df


def test_filename_parsing_functions():
    """Test string parsing utility functions."""
    via_tracks_path = "path/to/04.09.2023-01-Right-Loop05_tracks.csv"

    # Test video filename extraction
    video_name = via_tracks_to_video_filename(via_tracks_path)
    assert video_name == "04.09.2023-01-Right"

    # Test clip filename extraction
    clip_name = via_tracks_to_clip_filename(via_tracks_path)
    assert clip_name == "04.09.2023-01-Right-Loop05.mp4"

    # Test clip ID extraction
    clip_id = clip_filename_to_clip_id(clip_name)
    assert clip_id == "Loop05"


def test_group_files_per_video(tmp_path):
    """Test grouping of VIA track files by video."""
    # Create mock directory structure
    list_via_tracks_files = [
        "04.09.2023-01-Right-Loop01_tracks.csv",
        "04.09.2023-01-Right-Loop02_tracks.csv",
        "04.09.2023-02-Right-Loop01_tracks.csv",
    ]
    for f in list_via_tracks_files:
        (tmp_path / f).touch()

    # Group files
    map_videos_to_files = group_files_per_video(
        tmp_path, "*.csv", via_tracks_to_video_filename
    )

    assert len(map_videos_to_files) == 2
    assert "04.09.2023-01-Right" in map_videos_to_files
    assert "04.09.2023-02-Right" in map_videos_to_files
    assert len(map_videos_to_files["04.09.2023-01-Right"]) == 2
    assert len(map_videos_to_files["04.09.2023-02-Right"]) == 1


def test_load_ds_chunked(
    sample_via_tracks_file_factory, sample_metadata_df_factory
):
    """Test loading VIA tracks with metadata as extended dataset."""
    # Sample VIA tracks file
    via_tracks_filename = "04.09.2023-01-Right-Loop05_tracks.csv"
    via_tracks_path = sample_via_tracks_file_factory(via_tracks_filename)

    # Define metadata df that contains the clip for the VIA tracks file
    clip_name = via_tracks_filename.replace("_tracks.csv", ".mp4")
    df_metadata = sample_metadata_df_factory(clip_name)

    # Load dataset for this clip chunked
    ds = load_ds_chunked(
        via_tracks_path,
        df_metadata,
        chunks={"time": 100, "space": -1, "individuals": -1, "clip_id": -1},
    )

    # Check escape_state is a data var
    assert "escape_state" in ds.data_vars

    # Check clip_id is added as a dimension
    assert "clip_id" in ds.dims

    # Check clip_id non-index coordinates
    assert "clip_first_frame_0idx" in ds.coords
    assert "clip_last_frame_0idx" in ds.coords
    assert "clip_escape_first_frame_0idx" in ds.coords
    assert "clip_escape_type" in ds.coords

    # Verify chunking
    assert ds.chunks is not None


def test_get_video_dataset(sample_via_tracks_file_factory):
    """Test definition of video dataset from multiple clips."""
    # Define input VIA track files
    video_id = "04.09.2023-01-Right"
    via_track_files = [
        sample_via_tracks_file_factory(f"{video_id}-Loop05_tracks.csv"),
        sample_via_tracks_file_factory(f"{video_id}-Loop06_tracks.csv"),
    ]

    # Build sample metadata dataframe including those files
    video_clip_names = [
        f.name.replace("_tracks.csv", ".mp4") for f in via_track_files
    ]
    df_metadata = pd.DataFrame(
        {
            "loop_clip_name": video_clip_names,
            "loop_START_frame_ffmpeg": [100, 201],  # 1-indexed
            "loop_END_frame_ffmpeg": [200, 300],  # 1-indexed
            "escape_START_frame_0_based_idx": [150] * 2,  # 0-indexed
            "escape_type": ["triggered", "spontaneous"],
            "video_name": [f"{video_id}.mov"] * 2,
            "fps": [30.0] * 2,
        }
    )

    # Get video dataset for the input VIA track files and metadata
    ds = get_video_dataset(video_id, via_track_files, df_metadata)

    # Check dimensions
    assert "clip_id" in ds.dims
    assert ds.dims["clip_id"] == len(via_track_files)

    # Check attributes
    assert ds.attrs["video_id"] == video_id
    assert ds.attrs["source_file"] == via_track_files
    assert "fps" in ds.attrs

    # Verify the data variables are dask arrays
    assert ds.chunks is not None
    for var in ds.data_vars:
        assert isinstance(ds[var].data, dask.array.Array)
