import pytest
from unittest.mock import patch, MagicMock, Mock
import argparse
import numpy as np
from pathlib import Path

from crabs.detection_tracking.inference_model import DetectorInference


@pytest.fixture
def mock_tracker():
    args = argparse.Namespace(
        vid_path='/dummy/video.mp4',
        max_age=1,
        min_hits=3,
        score_threshold=0.3,
        iou_threshold=0.5
    )
    tracker = DetectorInference(args)
    tracker.args = MagicMock()
    
    return tracker


def evaluate_mota(gt_boxes, tracked_boxes, iou_threshold, prev_frame_ids):
    return 1, 2, 3, 4, 5, 1.0  

@pytest.mark.parametrize("gt_boxes_list, tracked_boxes_list, expected_mota_values", [
    (
        [[[0, 0, 10, 10]], [[0, 0, 10, 10]]],
        [[[0, 0, 10, 10, 1]], [[0, 0, 10, 10, 1]]],
        [1.0, 1.0]
    ),
    (
        [[[0, 0, 20, 20]], [[10, 10, 30, 30]]], 
        [[[5, 5, 25, 25, 2]], [[15, 15, 35, 35, 2]]],
        [1.0, 1.0]
    )
])
def test_evaluate_tracking(mock_tracker, gt_boxes_list, tracked_boxes_list, expected_mota_values):
    iou_threshold = 0.1

    with patch('crabs.detection_tracking.tracking_utils.evaluate_mota', side_effect=evaluate_mota):
        with patch.object(mock_tracker, 'save_tracking_results_to_csv') as mock_save:
            mota_values = mock_tracker.evaluate_tracking(
                gt_boxes_list, tracked_boxes_list, iou_threshold
            )

            assert mota_values == expected_mota_values
            mock_save.assert_called_once()
            

@pytest.mark.parametrize("save_csv_and_frames, save_video, gt_dir, frame_number", [
    (True, True, True, 1),
    (True, False, False, 1),
    (False, True, True, 1),
    (False, False, False, 1),
])
def test_save_required_output(mock_tracker, save_csv_and_frames, save_video, gt_dir, frame_number):
    # Mock attributes
    mock_tracker.args.save_csv_and_frames = save_csv_and_frames
    mock_tracker.args.save_video = save_video
    mock_tracker.args.gt_dir = gt_dir
    mock_tracker.tracking_output_dir = Path('mock_tracking_output_dir')
    mock_tracker.video_file_root = Path('mock_video_file_root')
    mock_tracker.csv_writer = Mock()
    mock_tracker.gt_boxes_list = [[(0, 0, 10, 10, 1)]]

    # Create a mock VideoWriter object
    mock_video_writer = Mock()
    mock_tracker.out = mock_video_writer
    
    tracked_boxes = [[0, 0, 10, 10, 1]]
    frame = np.zeros((100, 100, 3), dtype=np.uint8)
    
    with patch('crabs.detection_tracking.inference_model.save_frame_and_csv', autospec=True) as mock_save_frame_and_csv, \
         patch('crabs.detection_tracking.inference_model.draw_bbox', autospec=True) as mock_draw_bbox:
        
        # Call the method
        mock_tracker.save_required_output(tracked_boxes, frame, frame_number)
        
        # Check if save_frame_and_csv was called
        if save_csv_and_frames:
            print(mock_save_frame_and_csv)
            mock_save_frame_and_csv.assert_called_once_with(
                mock_tracker.video_file_root,
                mock_tracker.tracking_output_dir,
                tracked_boxes,
                frame,
                frame_number,
                mock_tracker.csv_writer
            )
        else:
            mock_save_frame_and_csv.assert_not_called()
        
        # Check if the video frame was written
        if save_video:
            assert mock_video_writer.write.call_count == 1 if not gt_dir else 2
            mock_draw_bbox.assert_called()
        else:
            mock_video_writer.write.assert_not_called()



