import matplotlib
matplotlib.use('Agg')
import argparse
from pathlib import Path
from unittest.mock import Mock, patch

import numpy as np
import pytest

from crabs.detection_tracking.inference_model import DetectorInference


@pytest.fixture
def mock_tracker():
    args = argparse.Namespace(
        vid_path="/dummy/video.mp4",
        model_dir="/dummy/model_checkpoint.ckpt",  # Use a placeholder path
        max_age=1,
        min_hits=3,
        score_threshold=0.3,
        iou_threshold=0.5,
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


# def test_load_trained_model(mock_tracker):
#     # Mock torch.nn.Module and torch.load
#     mock_torch_module = MagicMock(spec=torch.nn.Module)
#     torch_load_mock = MagicMock(return_value=mock_torch_module)

#     # Patch torch.load to return the mocked torch.nn.Module instance
#     with patch('torch.load', torch_load_mock):
#         # Mock builtins.open to prevent file access
#         open_mock = MagicMock()
#         open_mock.return_value.__enter__.return_value.read.return_value = 'dummy_content'
#         with patch('builtins.open', open_mock):
#             # Mock os.path.exists to return True
#             with patch('os.path.exists', MagicMock(return_value=True)):
#                 # Call the method to load the trained model
#                 mock_tracker.load_trained_model()

#                 # Assertions to verify the expected behavior
#                 assert isinstance(mock_tracker.trained_model, torch.nn.Module), "Trained model should be an instance of torch.nn.Module"
#                 assert not mock_tracker.trained_model.training, "Trained model should be in evaluation mode (not training)"
