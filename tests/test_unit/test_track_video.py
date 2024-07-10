import tempfile
from argparse import Namespace
from pathlib import Path
from unittest.mock import MagicMock, mock_open, patch

import pytest

from crabs.tracker.track_video import Tracking


@pytest.fixture
def mock_args():
    temp_dir = tempfile.mkdtemp()

    return Namespace(
        config_file="/path/to/config.yaml",
        video_path="/path/to/video.mp4",
        trained_model_path="/path/to/model.ckpt",
        output_dir=temp_dir,
        device="cuda",
        gt_path=None,
        save_video=None,
        run_on_video_dir =None
    )


@patch(
    "builtins.open",
    new_callable=mock_open,
    read_data="max_age: 10\nmin_hits: 3\niou_threshold: 0.1",
)
@patch("yaml.safe_load")
@patch("cv2.VideoCapture")
@patch("crabs.tracker.track_video.FasterRCNN.load_from_checkpoint")
@patch("crabs.tracker.track_video.Sort")
def test_tracking_setup(
    mock_sort,
    mock_load_from_checkpoint,
    mock_videocapture,
    mock_yaml_load,
    mock_open,
    mock_args,
):
    mock_yaml_load.return_value = {
        "max_age": 10,
        "min_hits": 3,
        "iou_threshold": 0.1,
    }

    mock_model = MagicMock()
    mock_load_from_checkpoint.return_value = mock_model

    mock_video_capture = MagicMock()
    mock_video_capture.isOpened.return_value = True
    mock_videocapture.return_value = mock_video_capture

    tracker = Tracking(mock_args)

    assert tracker.args.output_dir == mock_args.output_dir

    Path(mock_args.output_dir).rmdir()
