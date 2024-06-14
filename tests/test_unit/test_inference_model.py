import argparse
from pathlib import Path
from unittest.mock import MagicMock, Mock, patch

import numpy as np
import pytest
import torch

from crabs.detection_tracking.inference_model import DetectorInference
from crabs.detection_tracking.models import FasterRCNN


@pytest.fixture
def mock_tracker():
    args = argparse.Namespace(
        vid_path="/dummy/video.mp4",
        model_dir="/dummy/model_checkpoint.ckpt",  # Use a placeholder path
        max_age=1,
        min_hits=3,
        score_threshold=0.1,
        iou_threshold=0.1,
    )
    tracker = DetectorInference(args)
    tracker.args = args

    return tracker


def evaluate_mota(gt_boxes, tracked_boxes, iou_threshold, prev_frame_ids):
    return 1, 2, 3, 4, 5, 1.0


@pytest.mark.parametrize(
    "gt_boxes_list, tracked_boxes_list, expected_mota_values",
    [
        (
            [[[0, 0, 10, 10]], [[0, 0, 10, 10]]],
            [[[0, 0, 10, 10, 1]], [[0, 0, 10, 10, 1]]],
            [1.0, 1.0],
        ),
        (
            [[[0, 0, 20, 20]], [[10, 10, 30, 30]]],
            [[[5, 5, 25, 25, 2]], [[15, 15, 35, 35, 2]]],
            [1.0, 1.0],
        ),
    ],
)
def test_evaluate_tracking(
    mock_tracker, gt_boxes_list, tracked_boxes_list, expected_mota_values
):
    iou_threshold = 0.1

    with patch(
        "crabs.detection_tracking.tracking_utils.evaluate_mota",
        side_effect=evaluate_mota,
    ):
        with patch.object(
            mock_tracker, "save_tracking_results_to_csv"
        ) as mock_save:
            mota_values = mock_tracker.evaluate_tracking(
                gt_boxes_list, tracked_boxes_list, iou_threshold
            )

            assert mota_values == expected_mota_values
            mock_save.assert_called_once()


@pytest.mark.parametrize(
    "save_csv_and_frames, save_video, gt_dir, frame_number",
    [
        (True, True, True, 1),
        (True, False, False, 1),
        (False, True, True, 1),
        (False, False, False, 1),
    ],
)
def test_save_required_output(
    mock_tracker, save_csv_and_frames, save_video, gt_dir, frame_number
):
    mock_tracker.args.save_csv_and_frames = save_csv_and_frames
    mock_tracker.args.save_video = save_video
    mock_tracker.args.gt_dir = gt_dir
    mock_tracker.tracking_output_dir = Path("mock_tracking_output_dir")
    mock_tracker.video_file_root = Path("mock_video_file_root")
    mock_tracker.csv_writer = Mock()
    mock_tracker.gt_boxes_list = [[(0, 0, 10, 10, 1)]]

    mock_video_writer = Mock()
    mock_tracker.out = mock_video_writer

    tracked_boxes = [[0, 0, 10, 10, 1]]
    frame = np.zeros((100, 100, 3), dtype=np.uint8)

    with patch(
        "crabs.detection_tracking.inference_model.save_frame_and_csv",
        autospec=True,
    ) as mock_save_frame_and_csv, patch(
        "crabs.detection_tracking.inference_model.draw_bbox", autospec=True
    ) as mock_draw_bbox:
        mock_tracker.save_required_output(tracked_boxes, frame, frame_number)

        if save_csv_and_frames:
            print(mock_save_frame_and_csv)
            mock_save_frame_and_csv.assert_called_once_with(
                mock_tracker.video_file_root,
                mock_tracker.tracking_output_dir,
                tracked_boxes,
                frame,
                frame_number,
                mock_tracker.csv_writer,
            )
        else:
            mock_save_frame_and_csv.assert_not_called()

        if save_video:
            assert mock_video_writer.write.call_count == 1 if not gt_dir else 2
            mock_draw_bbox.assert_called()
        else:
            mock_video_writer.write.assert_not_called()


@pytest.fixture
def mock_trained_model():
    mock_model = MagicMock(spec=FasterRCNN)
    mock_model.training = False
    return mock_model


def test_load_trained_model(mock_tracker, mock_trained_model):
    with patch.object(
        FasterRCNN, "load_from_checkpoint", return_value=mock_trained_model
    ):
        mock_tracker.load_trained_model()

        assert isinstance(
            mock_tracker.trained_model, FasterRCNN
        ), "Trained model should be an instance of FasterRCNN"
        assert (
            not mock_tracker.trained_model.training
        ), "Trained model should be in evaluation mode (not training)"


@pytest.mark.parametrize(
    "prediction, expected_output",
    [
        # Test case 1: One box above threshold, one below
        (
            [
                {
                    "boxes": torch.tensor(
                        [[10, 20, 30, 40], [50, 60, 70, 80]]
                    ),
                    "scores": torch.tensor([0.8, 0.05]),
                    "labels": torch.tensor([1, 2]),
                }
            ],
            np.array([[10.0, 20.0, 30.0, 40.0, 0.8]]),
        ),
        # Test case 2: All boxes above threshold
        (
            [
                {
                    "boxes": torch.tensor(
                        [[10, 20, 30, 40], [50, 60, 70, 80]]
                    ),
                    "scores": torch.tensor([0.8, 0.4]),
                    "labels": torch.tensor([1, 2]),
                }
            ],
            np.array(
                [[10.0, 20.0, 30.0, 40.0, 0.8], [50.0, 60.0, 70.0, 80.0, 0.4]]
            ),
        ),
        # Test case 3: All boxes below threshold
        (
            [
                {
                    "boxes": torch.tensor(
                        [[10, 20, 30, 40], [50, 60, 70, 80]]
                    ),
                    "scores": torch.tensor([0.05, 0.05]),
                    "labels": torch.tensor([1, 2]),
                }
            ],
            np.array([]),
        ),
    ],
)
def test_prep_sort(mock_tracker, prediction, expected_output):
    output = mock_tracker.prep_sort(prediction)

    assert np.allclose(
        output, expected_output
    ), "The output of prep_sort is not as expected."

    assert (
        output.shape == expected_output.shape
    ), "The shape of the output is incorrect."
    assert isinstance(output, np.ndarray), "The output type is incorrect."


@pytest.mark.parametrize("save_video", [True, False])
def test_load_video(mock_tracker, save_video):
    mock_tracker.args.save_video = save_video

    mock_video_capture = MagicMock()
    mock_video_capture.isOpened.return_value = True
    mock_video_capture.get.side_effect = [640, 480, 30]

    mock_video_writer = MagicMock()

    with patch(
        "cv2.VideoCapture", return_value=mock_video_capture
    ) as mock_capture:
        with patch("cv2.VideoWriter", return_value=mock_video_writer):
            mock_tracker.load_video()

            mock_capture.assert_called_once_with("/dummy/video.mp4")

            mock_video_capture.isOpened.assert_called_once()
            if save_video:
                assert (
                    mock_video_capture.get.call_count == 3
                ), "get method not called 3 times"
            else:
                mock_video_capture.get.assert_not_called()
                mock_video_writer.assert_not_called()


@pytest.mark.parametrize("save_video", [True, False])
def test_load_video_failure(mock_tracker, save_video):
    # Set save_video parameter in the tracker instance
    mock_tracker.args.save_video = save_video

    # Mock for cv2.VideoCapture
    mock_video_capture = MagicMock()
    mock_video_capture.isOpened.return_value = False

    with patch("cv2.VideoCapture", return_value=mock_video_capture):
        with pytest.raises(Exception, match="Error opening video file"):
            mock_tracker.load_video()
