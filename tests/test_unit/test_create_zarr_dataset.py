import re
from contextlib import nullcontext as does_not_raise
from pathlib import Path

import pandas as pd
import pytest

from crabs.utils.create_zarr_dataset import (
    _clip_filename_to_clip_id,
    _get_video_fps,
    _group_files_per_video,
    _via_tracks_to_clip_filename,
    _via_tracks_to_video_filename,
    load_extended_ds,
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
    """Return as factory of metadata dataframes."""

    def _sample_metadata_df(list_clip_names):
        """Create a mock metadata dataframe for the input VIA track files."""
        n_clips = len(list_clip_names)
        list_video_names = [
            re.sub(r"-Loop\d+_tracks\.csv$", "", f.name) + ".mov"
            for f in list_clip_names
        ]
        return pd.DataFrame(
            {
                "loop_clip_name": list_clip_names,
                "loop_START_frame_ffmpeg": [100] * n_clips,  # 1-indexed
                "loop_END_frame_ffmpeg": [200] * n_clips,  # 1-indexed
                "escape_START_frame_0_based_idx": [150] * n_clips,  # 0-indexed
                "escape_type": ["spontaneous"] * n_clips,
                "video_name": list_video_names,
                "fps": [30.0] * n_clips,
            }
        )

    return _sample_metadata_df


def test_filename_parsing_functions():
    """Test string parsing utility functions."""
    via_tracks_path = "path/to/04.09.2023-01-Right-Loop05_tracks.csv"

    # Test video filename extraction
    video_name = _via_tracks_to_video_filename(via_tracks_path)
    assert video_name == "04.09.2023-01-Right"

    # Test clip filename extraction
    clip_name = _via_tracks_to_clip_filename(via_tracks_path)
    assert clip_name == "04.09.2023-01-Right-Loop05.mp4"

    # Test clip ID extraction
    clip_id = _clip_filename_to_clip_id(clip_name)
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
    map_videos_to_files = _group_files_per_video(
        tmp_path, "*.csv", _via_tracks_to_video_filename
    )

    assert len(map_videos_to_files) == 2
    assert "04.09.2023-01-Right" in map_videos_to_files
    assert "04.09.2023-02-Right" in map_videos_to_files
    assert len(map_videos_to_files["04.09.2023-01-Right"]) == 2
    assert len(map_videos_to_files["04.09.2023-02-Right"]) == 1


def test_load_extended_ds(
    sample_via_tracks_file_factory, sample_metadata_df_factory
):
    """Test loading VIA tracks with metadata as a `movement` dataset."""
    # Sample VIA tracks file
    via_tracks_filename = "04.09.2023-01-Right-Loop05_tracks.csv"
    via_tracks_path = sample_via_tracks_file_factory(via_tracks_filename)

    # Define metadata df that contains the clip for the VIA tracks file
    clip_name = via_tracks_filename.replace("_tracks.csv", ".mp4")
    df_metadata = sample_metadata_df_factory(clip_name)

    # Load dataset for this clip chunked
    ds = load_extended_ds(via_tracks_path, df_metadata)

    # Check escape_state is a data var
    assert "escape_state" in ds.data_vars

    # Check clip_id is added as a dimension
    assert "clip_id" in ds.dims

    # Check clip_id non-index coordinates
    assert "clip_first_frame_0idx" in ds.coords
    assert "clip_last_frame_0idx" in ds.coords
    assert "clip_escape_first_frame_0idx" in ds.coords
    assert "clip_escape_type" in ds.coords


@pytest.mark.parametrize(
    "list_files, expected_exception",
    [
        (
            [
                "04.09.2023-01-Right-Loop00_tracks.csv",
                "05.09.2023-01-Right-Loop01_tracks.csv",
            ],
            does_not_raise(),
        ),  # each row has a different video and different fps - this is OK
        (
            [
                "04.09.2023-01-Right-Loop00_tracks.csv",
                "04.09.2023-01-Right-Loop01_tracks.csv",
            ],
            pytest.raises(ValueError),
        ),  # both rows come from the same video, should be same fps
    ],
)
def test_get_video_fps(
    list_files,
    expected_exception,
    sample_via_tracks_file_factory,
    sample_metadata_df_factory,
):
    # Build mock metadata dataframe
    list_paths_to_files = [
        sample_via_tracks_file_factory(f) for f in list_files
    ]
    df_metadata = sample_metadata_df_factory(list_paths_to_files)

    # Assign different fps to first and second file
    df_metadata.loc[0, "fps"] = 30.0
    df_metadata.loc[1, "fps"] = 25.0

    # Get video_id from the first file
    video_id = _via_tracks_to_video_filename(list_paths_to_files[0])

    # Extract fps for that video_id
    with expected_exception as exc_info:
        fps = _get_video_fps(video_id, df_metadata)

    if not exc_info:
        assert fps == pytest.approx(30.0)
    else:
        assert f"Expected uniform fps for video '{video_id}'" in str(
            exc_info.value
        )


def test_create_temp_zarr_store():
    pass


def test_create_final_zarr_store():
    pass
