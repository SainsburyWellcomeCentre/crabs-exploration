from pathlib import Path
from unittest.mock import ANY, Mock, patch

import numpy as np
import pytest

from crabs.tracker.utils.io import save_required_output


@pytest.fixture
def setup_mocks():
    csv_writer_mock = Mock()
    video_output_mock = Mock()

    video_file_root = Path("sample_video")
    tracking_output_dir = Path("output")
    tracked_boxes = [[10, 20, 30, 40, 1]]
    frame = np.zeros((480, 640, 3), dtype=np.uint8)
    frame_number = 1
    pred_score = [[0.9]]

    return (
        video_file_root,
        tracking_output_dir,
        csv_writer_mock,
        video_output_mock,
        tracked_boxes,
        frame,
        frame_number,
        pred_score,
    )


@patch("crabs.tracker.utils.io.save_output_frames")
@patch("crabs.tracker.utils.io.write_tracked_bbox_to_csv")
@patch("crabs.tracker.utils.io.draw_bbox")
@pytest.mark.parametrize(
    "save_frames, save_video",
    [
        (True, True),
        (True, False),
        (False, True),
        (False, False),
    ],
)
def test_save_required_output(
    mock_draw_bbox,
    mock_write_tracked_bbox_to_csv,
    mock_save_frames,
    save_frames,
    save_video,
    setup_mocks,
):
    (
        video_file_root,
        tracking_output_dir,
        csv_writer,
        video_output,
        tracked_boxes,
        frame,
        frame_number,
        pred_score,
    ) = setup_mocks

    save_required_output(
        video_file_root,
        save_frames,
        tracking_output_dir,
        csv_writer,
        save_video,
        video_output,
        tracked_boxes,
        frame,
        frame_number,
        pred_score,
    )

    frame_name = f"{video_file_root}_frame_{frame_number:08d}.png"

    mock_write_tracked_bbox_to_csv.assert_called_once_with(
        tracked_boxes[0], frame, frame_name, csv_writer, pred_score[0]
    )

    if save_frames:
        mock_save_frames.assert_called_once_with(
            frame_name,
            tracking_output_dir,
            frame,
            frame_number,
        )

    if save_video:
        mock_draw_bbox.assert_any_call(
            ANY, (10, 20), (30, 40), (0, 0, 255), "id : 1"
        )
        video_output.write.assert_called_once_with(ANY)
    else:
        mock_draw_bbox.assert_not_called()
        video_output.write.assert_not_called()
