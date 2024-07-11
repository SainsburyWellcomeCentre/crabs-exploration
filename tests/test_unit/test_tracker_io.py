from pathlib import Path
from unittest.mock import MagicMock, patch

import numpy as np
import pytest

from crabs.tracker.utils.io import save_required_output


@pytest.mark.parametrize(
    "save_frames, save_video, ground_truth_dict",
    [
        (
            True,
            True,
            {
                1: {
                    "bbox": [[15, 25, 35, 45], [55, 65, 75, 85]],
                    "id": [101, 102],
                }
            },
        ),
        (True, False, {}),
        (
            False,
            True,
            {
                1: {
                    "bbox": [[15, 25, 35, 45], [55, 65, 75, 85]],
                    "id": [101, 102],
                }
            },
        ),
        (False, False, {}),
    ],
)
def test_save_required_output(save_frames, save_video, ground_truth_dict):
    with patch(
        "crabs.tracker.utils.io.write_tracked_bbox_to_csv"
    ) as mock_write_tracked_bbox_to_csv, patch(
        "crabs.tracker.utils.io.save_output_frames"
    ) as mock_save_output_frames:
        video_output_mock = MagicMock()
        csv_writer_mock = MagicMock()

        video_file_root = Path("/path/to/video")
        tracking_output_dir = Path("/path/to/output")
        tracked_boxes = [[10, 20, 30, 40, 1], [50, 60, 70, 80, 2]]
        frame = np.zeros((480, 640, 3), dtype=np.uint8)
        frame_number = 1
        pred_scores = np.array([0.9, 0.8])

        save_required_output(
            video_file_root,
            save_frames,
            tracking_output_dir,
            csv_writer_mock,
            save_video,
            video_output_mock,
            tracked_boxes,
            frame,
            frame_number,
            pred_scores,
            ground_truth_dict,
        )

        for bbox, pred_score in zip(tracked_boxes, pred_scores):
            mock_write_tracked_bbox_to_csv.assert_any_call(
                bbox,
                frame,
                f"{video_file_root}_frame_{frame_number:08d}.png",
                csv_writer_mock,
                pred_score,
            )
        if save_frames:
            mock_save_output_frames.assert_called_once_with(
                f"{video_file_root}_frame_{frame_number:08d}.png",
                tracking_output_dir,
                frame,
                frame_number,
            )
        else:
            mock_save_output_frames.assert_not_called()

        if ground_truth_dict:
            assert video_output_mock.write.call_count == 1
        if save_video:
            assert video_output_mock.write.call_count == 1
        else:
            assert video_output_mock.write.call_count == 0
