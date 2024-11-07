import tempfile
from argparse import Namespace
from pathlib import Path
from unittest.mock import MagicMock, mock_open, patch

import pytest

from crabs.tracker.track_video import Tracking


@pytest.fixture
def mock_args():
    tmp_dir = tempfile.mkdtemp()

    return Namespace(
        config_file="/path/to/config.yaml",
        video_path="/path/to/video.mp4",
        trained_model_path="path/to/model.ckpt",
        output_dir=tmp_dir,
        accelerator="gpu",
        annotations_file=None,
        save_video=None,
        save_frames=None,
    )


@patch(
    "builtins.open",
    new_callable=mock_open,
    read_data="max_age: 10\nmin_hits: 3\niou_threshold: 0.1",
)
@patch("cv2.VideoCapture")
@patch("crabs.tracker.utils.io.get_video_parameters")
@patch("crabs.tracker.track_video.get_config_from_ckpt")
@patch("crabs.tracker.track_video.get_mlflow_parameters_from_ckpt")
# we patch where the function is looked at, see
# https://docs.python.org/3/library/unittest.mock.html#where-to-patch
@patch("yaml.safe_load")
def test_tracking_constructor(
    mock_yaml_load,
    mock_get_mlflow_parameters_from_ckpt,
    mock_get_config_from_ckpt,
    mock_get_video_parameters,
    mock_videocapture,
    mock_open,
    mock_args,
):
    # mock reading tracking config from file
    mock_yaml_load.return_value = {
        "max_age": 10,
        "min_hits": 3,
        "iou_threshold": 0.1,
    }

    # mock getting mlflow parameters from checkpoint
    mock_get_mlflow_parameters_from_ckpt.return_value = {
        "run_name": "trained_model_run_name",
        "cli_args/experiment_name": "trained_model_expt_name",
    }

    # mock getting trained model's config
    mock_get_config_from_ckpt.return_value = {}

    # mock getting video parameters
    mock_get_video_parameters.return_value = {
        "total_frames": 614,
        "frame_width": 1920,
        "frame_height": 1080,
        "fps": 60,
    }

    # mock input video as if opened correctly
    mock_video_capture = MagicMock()
    mock_video_capture.isOpened.return_value = True
    mock_videocapture.return_value = mock_video_capture

    # instantiate tracking interface
    tracker = Tracking(mock_args)

    # check output dir is created correctly
    # TODO: add asserts for other attributes assigned in constructor
    assert tracker.args.output_dir == mock_args.output_dir

    # delete output dir
    Path(mock_args.output_dir).rmdir()
