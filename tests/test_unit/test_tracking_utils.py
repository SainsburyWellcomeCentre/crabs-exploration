import csv
import io
from pathlib import Path

import numpy as np
import pytest

from crabs.tracker.utils.tracking import (
    extract_bounding_box_info,
    get_ground_truth_data,
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
    pred_score = 0.900

    write_tracked_bbox_to_csv(bbox, frame, frame_name, csv_writer, pred_score)

    expected_row = (
        "frame_0001.png",
        30000,
        '"{""clip"":123}"',
        1,
        0,
        '"{""name"":""rect"",""x"":10,""y"":20,""width"":40,""height"":60}"',
        '"{""track"":""1"", ""confidence"":""0.9""}"',
    )
    expected_row_str = ",".join(map(str, expected_row))
    assert csv_output.getvalue().strip() == expected_row_str


@pytest.fixture
def gt_path():
    test_csv_file = Path(__file__).parents[1] / "data" / "gt_test.csv"
    return test_csv_file


def test_get_ground_truth_data(gt_path):
    ground_truth_dict = get_ground_truth_data(gt_path)

    assert isinstance(ground_truth_dict, dict)
    assert all(
        isinstance(frame_data, dict)
        for frame_data in ground_truth_dict.values()
    )

    for frame_number, data in ground_truth_dict.items():
        assert isinstance(frame_number, int)
        assert isinstance(data["bbox"], np.ndarray)
        assert isinstance(data["id"], np.ndarray)
        assert data["bbox"].shape[1] == 4


def test_ground_truth_data_from_csv(gt_path):
    expected_data = {
        10: {
            "bbox": np.array(
                [
                    [2894.8606, 975.8517, 2945.8606, 1016.8517],
                    [940.6089, 1192.637, 989.6089, 1230.637],
                ],
                dtype=np.float32,
            ),
            "id": np.array([2.0, 1.0], dtype=np.float32),
        },
        20: {
            "bbox": np.array(
                [[940.6089, 1192.637, 989.6089, 1230.637]], dtype=np.float32
            ),
            "id": np.array([2.0], dtype=np.float32),
        },
    }

    ground_truth_dict = get_ground_truth_data(gt_path)

    for frame_number, expected_frame_data in expected_data.items():
        assert frame_number in ground_truth_dict

        assert len(ground_truth_dict[frame_number]["bbox"]) == len(
            expected_frame_data["bbox"]
        )
        for bbox, expected_bbox in zip(
            ground_truth_dict[frame_number]["bbox"],
            expected_frame_data["bbox"],
        ):
            assert np.allclose(
                bbox, expected_bbox
            ), f"Frame {frame_number}, bbox mismatch"

        assert np.array_equal(
            ground_truth_dict[frame_number]["id"], expected_frame_data["id"]
        ), f"Frame {frame_number}, id mismatch"
