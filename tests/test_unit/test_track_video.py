import re
from argparse import Namespace
from pathlib import Path
from typing import Callable

import pytest
import yaml

from crabs.tracker.track_video import Tracking


@pytest.fixture()
def create_tracking_config_file():
    """Return a factory to create a tracking config file under a Pytest
    temporary directory.
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

    return _create_tracking_config_file


@pytest.fixture()
def create_mock_args():
    """Return a factory of mock arguments for Tracking class.

    The factory returns a Namespace object with mock arguments for Tracking
    class. When called, the factory also creates a temporary file with the
    specified tracking config, whose path gets added to the Namespace object.
    """

    def _create_mock_args(
        args_dict: dict,
    ) -> Namespace:
        """Return a Namespace object with mock arguments for Tracking class,
        given a tracking config dictionary and a factory to create a tracking
        config files.
        """
        return Namespace(**args_dict)

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
    pathlib_mkdir = Path.mkdir

    def mock_mkdir(self, parents=False, exist_ok=False):
        return pathlib_mkdir(
            tmp_path / self, parents=parents, exist_ok=exist_ok
        )

    monkeypatch.setattr(Path, "mkdir", mock_mkdir)


def test_tracking_constructor(
    create_mock_args: Callable,
    create_tracking_config_file: Callable,
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
    mock_args = create_mock_args(
        {
            "video_path": "/path/to/video.mp4",
            "trained_model_path": "path/to/model.ckpt",
            "config_file": create_tracking_config_file(
                tracking_config, tmp_path
            ),
            "accelerator": "gpu",
            "output_dir": "tracking_output",
            "output_dir_no_timestamp": False,
            "annotations_file": None,
            "save_video": False,
            "save_frames": False,
        }
    )

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


@pytest.mark.parametrize(
    "output_dir",
    [
        "tracking_output",  # default
        "output",
    ],
)
@pytest.mark.parametrize(
    "output_dir_no_timestamp",
    [
        False,
        True,
    ],
)
@pytest.mark.parametrize(
    "save_video",
    [
        False,
        True,
    ],
)
@pytest.mark.parametrize(
    "save_frames",
    [
        False,
        True,
    ],
)
def test_prep_outputs(
    output_dir,
    output_dir_no_timestamp,
    save_video,
    save_frames,
    create_mock_args,
    create_tracking_config_file,
    mock_mkdir,
    tmp_path,
    monkeypatch,
):
    """Test attributes related to outputs are correctly defined.

    Checks paths for required outputs are defined, and that the output
    directory and (optional) frames subdirectory are created. The
    directories are created under a temporary directory created by pytest.

    """
    # Create mock arguments for the constructor
    mock_args = create_mock_args(
        {
            "video_path": "/path/to/video.mp4",
            "trained_model_path": "path/to/model.ckpt",
            "config_file": create_tracking_config_file({}, tmp_path),
            "accelerator": "gpu",
            "output_dir": output_dir,
            "output_dir_no_timestamp": output_dir_no_timestamp,
            "annotations_file": None,
            "save_video": save_video,
            "save_frames": save_frames,
        }
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

    # Instantiate tracking interface - includes prep_outputs step
    # Note: mkdir is patched via `mock_mkdir` to create any output
    # directories under a Pytest temporary directory
    tracker = Tracking(mock_args)

    # check name of output directory
    if mock_args.output_dir:
        output_dir_root = mock_args.output_dir
    else:
        output_dir_root = "tracking_output"  # default
    if mock_args.output_dir_no_timestamp:
        assert tracker.tracking_output_dir.stem == output_dir_root
    else:
        output_dir_regexp = re.compile(rf"{output_dir_root}_\d{{8}}_\d{{6}}$")
        assert output_dir_regexp.match(tracker.tracking_output_dir.stem)

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

    # check output directory is created
    # (under pytest temporary directory)
    assert (tmp_path / tracker.tracking_output_dir).exists()

    # check path to frames subdirectory is defined and created
    if mock_args.save_frames:
        # assert directory name
        assert tracker.frames_subdir == (
            tracker.tracking_output_dir
            / f"{tracker.input_video_file_root}_frames"
        )
        # assert creation
        assert (tmp_path / tracker.frames_subdir).exists()
