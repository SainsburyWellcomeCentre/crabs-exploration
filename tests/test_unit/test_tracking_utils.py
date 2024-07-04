import csv
import io

import numpy as np
import pytest

from crabs.tracker.utils.tracking import (
    extract_bounding_box_info,
    write_tracked_bbox_to_csv,
)


def test_extract_bounding_box_info():
    csv_row = [
        "frame_00000001.png",
        "26542080",
        "{" "clip" ":123}",
        "1",
        "0",
        '{"name":"rect","x":2894.860594987354,"y":975.8516839863181,"width":51,"height":41}',
        '{"track":"79.0"}',
    ]

    result = extract_bounding_box_info(csv_row)

    expected_result = {
        "frame_number": 0,
        "x": 2894.860594987354,
        "y": 975.8516839863181,
        "width": 51,
        "height": 41,
        "id": "79.0",
    }

    assert result == expected_result


@pytest.fixture
def csv_output():
    return io.StringIO()


@pytest.fixture
def csv_writer(csv_output):
    return csv.writer(csv_output)


def test_write_tracked_bbox_to_csv(csv_writer, csv_output):
    bbox = np.array([10, 20, 50, 80, 1])
    frame = np.zeros((100, 100, 3), dtype=np.uint8)
    frame_name = "frame_0001.png"
    theta = 1.0

    write_tracked_bbox_to_csv(bbox, frame, frame_name, csv_writer, theta)

    expected_row = (
        "frame_0001.png",
        30000,
        '"{""clip"":123}"',
        1,
        0,
        '"{""name"":""rect"",""x"":10,""y"":20,""width"":40,""height"":60}"',
        '"{""track"":""1"", ""theta"":""1.0""}"',
    )
    expected_row_str = ",".join(map(str, expected_row))
    assert csv_output.getvalue().strip() == expected_row_str
