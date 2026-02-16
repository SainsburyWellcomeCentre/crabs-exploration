import re
from contextlib import nullcontext as does_not_raise
from pathlib import Path

import pandas as pd
import pytest
import xarray as xr
import zarr

from crabs.utils.create_zarr_dataset import (
    _clip_filename_to_clip_id,
    _get_video_fps,
    _group_files_per_video,
    _via_tracks_to_clip_filename,
    _via_tracks_to_video_filename,
    create_final_zarr_store,
    create_temp_zarr_store,
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

    def _sample_metadata_df(list_via_track_filepaths):
        """Create a mock metadata dataframe for the input VIA track files."""
        list_clip_names = [
            Path(p).name.removesuffix("_tracks.csv") + ".mp4"
            for p in list_via_track_filepaths
        ]
        n_clips = len(list_clip_names)
        list_video_names = [
            re.sub(r"-Loop\d+_tracks\.csv$", "", Path(f).name) + ".mov"
            for f in list_via_track_filepaths
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
    # list_clip_filepath = [
    #     Path(str(via_tracks_path).replace("_tracks.csv", ".mp4"))
    # ]
    df_metadata = sample_metadata_df_factory([via_tracks_path])

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


def test_create_temp_zarr_store(
    tmp_path, sample_via_tracks_file_factory, sample_metadata_df_factory
):
    """Test creation of temporary zarr store from VIA tracks files."""
    # Create a set of VIA tracks files for two different videos
    list_via_paths = [
        sample_via_tracks_file_factory(fname)
        for fname in [
            "04.09.2023-01-Right-Loop00_tracks.csv",
            "04.09.2023-01-Right-Loop01_tracks.csv",
            "05.09.2023-01-Right-Loop00_tracks.csv",
            "05.09.2023-01-Right-Loop01_tracks.csv",
        ]
    ]

    # Create metadata CSV for those files
    df_metadata = sample_metadata_df_factory(list_via_paths)
    metadata_csv = tmp_path / "metadata.csv"
    df_metadata.to_csv(metadata_csv, index=False)

    # Define temp zarr store path
    temp_zarr_path = tmp_path / "temp_store.zarr"

    # Create temp zarr store
    temp_zarr_store_path, map_video_to_attrs = create_temp_zarr_store(
        temp_zarr_store=str(temp_zarr_path),
        temp_zarr_mode_store="w-",
        temp_zarr_mode_group="w-",
        via_tracks_dir=Path(list_via_paths[0]).parent,
        # all created in same dir
        via_tracks_glob_pattern="*_tracks.csv",
        # exclude metadata.csv!
        metadata_csv=metadata_csv,
    )

    # Check temporary zarr store structure
    assert temp_zarr_store_path.exists()
    root = zarr.open_group(temp_zarr_store_path, mode="r")
    all_clip_groups = [
        f"{video_id}/{clip_id}"
        for video_id in root.group_keys()
        for clip_id in root[video_id].group_keys()
    ]
    assert len(all_clip_groups) == 4  # one per VIA tracks file

    # Check video attributes
    list_expected_video_ids = ["04.09.2023-01-Right", "05.09.2023-01-Right"]
    assert list(map_video_to_attrs.keys()) == list_expected_video_ids
    assert all(
        sorted(map_video_to_attrs[video_id].keys()) == ["fps", "source_file"]
        for video_id in list_expected_video_ids
    )
    assert all(
        map_video_to_attrs[video_id]["fps"] == pytest.approx(30.0)
        for video_id in list_expected_video_ids
    )
    assert map_video_to_attrs["04.09.2023-01-Right"]["source_file"] == [
        f.as_posix() for f in list_via_paths[0:2]
    ]
    assert map_video_to_attrs["05.09.2023-01-Right"]["source_file"] == [
        f.as_posix() for f in list_via_paths[2:4]
    ]


def test_create_final_zarr_store(
    tmp_path, sample_via_tracks_file_factory, sample_metadata_df_factory
):
    """Test creation of final zarr store from temp zarr store."""
    # Create a set of VIA tracks files for two different videos
    list_via_paths = [
        sample_via_tracks_file_factory(fname)
        for fname in [
            "04.09.2023-01-Right-Loop00_tracks.csv",
            "04.09.2023-01-Right-Loop01_tracks.csv",
            "05.09.2023-01-Right-Loop00_tracks.csv",
        ]
    ]

    # Create metadata CSV for those files
    df_metadata = sample_metadata_df_factory(list_via_paths)
    metadata_csv = tmp_path / "metadata.csv"
    df_metadata.to_csv(metadata_csv, index=False)

    # Define temp and final zarr store paths
    temp_zarr_path = tmp_path / "temp_store.zarr"
    final_zarr_path = tmp_path / "final_store.zarr"

    # Create temp zarr store first
    temp_zarr_store_path, map_video_to_attrs = create_temp_zarr_store(
        temp_zarr_store=str(temp_zarr_path),
        temp_zarr_mode_store="w-",
        temp_zarr_mode_group="w-",
        via_tracks_dir=Path(list_via_paths[0]).parent,
        via_tracks_glob_pattern="*_tracks.csv",
        metadata_csv=metadata_csv,
    )

    # Create final zarr store from temp store
    create_final_zarr_store(
        temp_zarr_store=temp_zarr_store_path,
        map_video_to_attrs=map_video_to_attrs,
        zarr_store=final_zarr_path,
        zarr_mode_store="w-",
        zarr_mode_group="w-",
    )

    # Check final store and verify structure
    assert final_zarr_path.exists()
    dt = xr.open_datatree(final_zarr_path, engine="zarr")

    # Check groups correspond to videos
    list_expected_video_ids = ["04.09.2023-01-Right", "05.09.2023-01-Right"]
    assert sorted(dt.children.keys()) == sorted(list_expected_video_ids)

    # Check each video dataset contains expected clips
    ds_video_1 = dt["04.09.2023-01-Right"].to_dataset()
    assert ds_video_1.sizes["clip_id"] == 2  # Loop00 and Loop01

    ds_video_2 = dt["05.09.2023-01-Right"].to_dataset()
    assert ds_video_2.sizes["clip_id"] == 1  # Loop00 only

    # Check video attributes are added to dataset 1
    assert ds_video_1.attrs["video_id"] == "04.09.2023-01-Right"
    assert ds_video_1.attrs["fps"] == pytest.approx(30.0)
    assert "source_file" in ds_video_1.attrs

    # Check video attributes are added to dataset 2
    assert ds_video_2.attrs["video_id"] == "05.09.2023-01-Right"
    assert ds_video_2.attrs["fps"] == pytest.approx(30.0)
    assert "source_file" in ds_video_2.attrs


def test_main():
    pass
