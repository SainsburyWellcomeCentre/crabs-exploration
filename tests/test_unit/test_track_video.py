from argparse import Namespace
from pathlib import Path
from typing import Callable

import pytest
import yaml

from crabs.tracker.track_video import Tracking


@pytest.fixture()
def create_mock_args():
    """Return a factory of mock arguments for Tracking class.

    The factory returns a Namespace object with mock arguments for Tracking
    class. When called, the factory also creates a temporary file with the
    specified tracking config, whose path gets added to the Namespace object.
    """

    def _create_tracking_config_file(
        tracking_config: dict, tmp_path: Path
    ) -> Path:
        """Return the path to a tracking config file under a Pytest temporary
        directory.
        """
        path_to_config = Path(tmp_path) / "tracking_config.yaml"

        with open(path_to_config, "w") as outfile:
            yaml.dump(
                tracking_config,
                outfile,
            )
        return path_to_config

    def _create_mock_args(
        tracking_config: dict,
        tmp_path: Path,
    ) -> Namespace:
        """Return a Namespace object with mock arguments for Tracking class,
        given a tracking config dictionary and a factory to create a tracking
        config files.
        """
        return Namespace(
            video_path="/path/to/video.mp4",
            trained_model_path="path/to/model.ckpt",
            config_file=_create_tracking_config_file(
                tracking_config, tmp_path
            ),
            accelerator="gpu",
            output_dir="tracking_output",  # default
            output_dir_no_timestamp=None,
            annotations_file=None,
            save_video=None,
            save_frames=None,
        )

    return _create_mock_args


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
        return Path(tmp_path).mkdir(parents, exist_ok)

    monkeypatch.setattr(Path, "mkdir", mock_mkdir)


def test_tracking_constructor(
    create_mock_args: Callable,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
):
    """Test constructor for Tracking class."""
    # Create mock arguments for the constructor
    tracking_config = {
        "max_age": 10,
        "min_hits": 3,
        "iou_threshold": 0.1,
    }
    mock_args = create_mock_args(tracking_config, tmp_path)

    # mock getting mlflow parameters from checkpoint
    trained_model_mlflow_params = {
        "run_name": "trained_model_run_name",
        "cli_args/experiment_name": "trained_model_expt_name",
    }
    monkeypatch.setattr(
        "crabs.tracker.track_video.get_mlflow_parameters_from_ckpt",
        lambda x: trained_model_mlflow_params,
    )

    # mock getting trained model's config
    trained_model_config: dict = {}
    monkeypatch.setattr(
        "crabs.tracker.track_video.get_config_from_ckpt",
        lambda **kwargs: trained_model_config,
    )

    # mock prep_outputs method
    monkeypatch.setattr(
        "crabs.tracker.track_video.Tracking.prep_outputs",
        lambda x: None,
    )

    # instantiate the Tracking class
    tracker = Tracking(mock_args)

    # check attributes from constructor are correctly defined
    assert tracker.args == mock_args
    assert tracker.config_file == mock_args.config_file
    assert tracker.config == tracking_config
    assert tracker.trained_model_path == mock_args.trained_model_path
    assert (
        tracker.trained_model_run_name
        == trained_model_mlflow_params["run_name"]
    )
    assert (
        tracker.trained_model_expt_name
        == trained_model_mlflow_params["cli_args/experiment_name"]
    )
    assert tracker.trained_model_config == trained_model_config
    assert tracker.input_video_path == mock_args.video_path
    assert tracker.input_video_file_root == Path(mock_args.video_path).stem
    assert tracker.tracking_output_dir_root == mock_args.output_dir
    assert tracker.frame_name_format_str == "frame_{frame_idx:08d}.png"
    assert tracker.accelerator == "cuda"


def test_prep_outputs(mock_args, mock_mkdir, mock_tracker_init_requirements):
    """Test attributes related to outputs are correctly defined.

    Check paths are defined and output directory and
    optionally frames subdirectory are created.

    """
    # Instantiate tracking interface
    # mkdir should be patched to create output directory
    # under a Pytest temporary directory
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
