from argparse import Namespace
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from crabs.tracker.track_video import Tracking


@pytest.fixture()
def mock_args(tmp_path: Path) -> Namespace:
    return Namespace(
        video_path="/path/to/video.mp4",
        trained_model_path="path/to/model.ckpt",
        accelerator="gpu",
        output_dir=str(tmp_path),
        output_dir_no_timestamp=None,
        annotations_file=None,
        save_video=None,
        save_frames=None,
    )


@pytest.fixture()
def mock_mkdir(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    """Monkeypatch pathlib.Path.mkdir().

    Instead of creating the directory at the path specified,
    mock the method to create the directory under a temporary
    directory created by pytest.

    Parameters
    ----------
    tmp_path : pathlib.Path
        a temporary directory created by pytest
    monkeypatch : pytest.MonkeyPatch
        a monkeypatch fixture

    """

    # monkeypatch Path.mkdir() to create directories under
    # a temporary directory created by pytest
    def mock_mkdir(parents, exist_ok):
        return Path(tmp_path).mkdir(parents=parents, exist_ok=exist_ok)

    monkeypatch.setattr(Path, "mkdir", mock_mkdir)


def test_tracking_constructor(monkeypatch, mock_args, tmp_path):
    # Mock the open function to use a mock_open object
    mock_open = MagicMock()
    monkeypatch.setattr("builtins.open", mock_open)

    # mock reading tracking config from file
    monkeypatch.setattr(
        "yaml.safe_load",
        lambda x: {
            "max_age": 10,
            "min_hits": 3,
            "iou_threshold": 0.1,
        },
    )

    # mock getting mlflow parameters from checkpoint
    monkeypatch.setattr(
        "crabs.tracker.track_video.get_mlflow_parameters_from_ckpt",
        lambda x: {
            "run_name": "trained_model_run_name",
            "cli_args/experiment_name": "trained_model_expt_name",
        },
    )

    # mock getting trained model's config
    monkeypatch.setattr(
        "crabs.tracker.track_video.get_config_from_ckpt",
        lambda **kwargs: {},
    )

    # Create a temporary directory for the output
    mock_args.output_dir = str(tmp_path)

    # Instantiate the Tracking class
    tracker = Tracking(mock_args)

    # Check output dir is created correctly
    assert tracker.args.output_dir == mock_args.output_dir

    # Additional assertions and checks can be added here

    # Clean up the temporary directory
    Path(mock_args.output_dir).rmdir()


@patch("yaml.safe_load")
def test_prep_outputs(mock_args, mock_mkdir):
    """Test attributes related to outputs are correctly defined.

    Check paths are defined and output directory and
    optionally frames subdirectory are created.

    """
    # Instantiate tracking interface
    tracker = Tracking(mock_args)

    # Run prep_outputs method
    tracker.prep_outputs()

    # assert output directory is created
    # can I mock current working directory?
    # mock mkdir to create everything under a pytest tempdir?
    # - with default name
    # - with required name
    # - with or without timestamp
    if mock_args.output_dir_no_timestamp:
        if mock_args.output_dir:
            assert (
                Path(tracker.tracking_output_dir).stem == mock_args.output_dir
            )
        else:
            assert Path(tracker.tracking_output_dir).stem == "tracking_output"

    assert tracker.tracking_output_dir.exists()

    # check path to csv file with detections is defined
    assert tracker.csv_file_path == str(
        tracker.tracking_output_dir
        / f"{tracker.input_video_file_root}_tracks.csv"
    )

    # check path to output video is defined
    if mock_args.save_video:
        assert tracker.output_video_path == str(
            tracker.tracking_output_dir
            / f"{tracker.input_video_file_root}_tracks.mp4"
        )

    # check path to frames subdirectory is defined and created
    if mock_args.save_frames:
        assert tracker.frames_subdir == str(
            tracker.tracking_output_dir
            / f"{tracker.input_video_file_root}_frames"
        )
        assert tracker.frames_subdir.exists()
