import csv

import numpy as np

from crabs.tracker.utils.io import write_tracked_detections_to_csv


def test_write_tracked_detections_to_csv(tmp_path):
    # Create test data
    csv_file_path = tmp_path / "test_output.csv"

    # Create dictionary with tracked bounding boxes for 2 frames
    tracked_bboxes_dict = {}

    # frame_idx = 0
    tracked_bboxes_dict[0] = {
        "tracked_boxes": np.array([[10, 20, 30, 40], [50, 60, 70, 80]]),
        "ids": np.array([1, 2]),
        "scores": np.array([0.9, 0.8]),
    }

    # frame_idx = 1
    tracked_bboxes_dict[1] = {
        "tracked_boxes": np.array([[15, 25, 35, 45]]),
        "ids": np.array([1]),
        "scores": np.array([0.85]),
    }
    frame_name_regexp = "frame_{frame_idx:08d}.png"
    all_frames_size = 8888

    # Call function
    write_tracked_detections_to_csv(
        csv_file_path,
        tracked_bboxes_dict,
        frame_name_regexp,
        all_frames_size,
    )

    # Read csv file
    with open(csv_file_path, newline="") as csvfile:
        csv_reader = csv.reader(csvfile)
        rows = list(csv_reader)

    # Expected header
    expected_header = [
        "filename",
        "file_size",
        "file_attributes",
        "region_count",
        "region_id",
        "region_shape_attributes",
        "region_attributes",
    ]

    # Expected rows
    expected_rows = [
        expected_header,
        [
            "frame_00000000.png",
            "8888",
            '{"clip":123}',
            "1",
            "0",
            '{"name":"rect","x":10,"y":20,"width":20,"height":20}',
            '{"track":"1", "confidence":"0.9"}',
        ],
        [
            "frame_00000000.png",
            "8888",
            '{"clip":123}',
            "1",
            "0",
            '{"name":"rect","x":50,"y":60,"width":20,"height":20}',
            '{"track":"2", "confidence":"0.8"}',
        ],
        [
            "frame_00000001.png",
            "8888",
            '{"clip":123}',
            "1",
            "0",
            '{"name":"rect","x":15,"y":25,"width":20,"height":20}',
            '{"track":"1", "confidence":"0.85"}',
        ],
    ]

    # Assert the header
    assert rows[0] == expected_header

    # Assert the rows
    for i, expected_row in enumerate(expected_rows[1:], start=1):
        assert rows[i] == expected_row
