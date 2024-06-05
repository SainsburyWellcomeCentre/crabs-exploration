import csv
import io
from pathlib import Path

import numpy as np
import pytest

from crabs.detection_tracking.tracking_utils import (
    calculate_iou,
    count_identity_switches,
    create_gt_list,
    evaluate_mota,
    extract_bounding_box_info,
    get_ground_truth_data,
    write_tracked_bbox_to_csv,
)


@pytest.mark.parametrize(
    "prev_frame_id, current_frame_id, n_gt, expected_output",
    [
        (None, [[6, 5, 4, 3, 2, 1]], 6, 0),
        ([[6, 5, 4, 3, 2, 1]], [[6, 5, 4, 3, 2, 1]], 6, 0),
        ([[5, 6, 4, 3, 1, 2]], [[6, 5, 4, 3, 2, 1]], 6, 0),
        ([[6, 5, 4, 3, 2, 1]], [[6, 5, 4, 2, 1]], 5, 0),
        ([[6, 5, 4, 3, 2, 1]], [[6, 5, 4, 2, 1, 7]], 6, 1),
        ([[6, 5, 4, 2, 1]], [[6, 5, 4, 2, 7]], 5, 1),
        ([[6, 5, 4, 3, 2]], [[6, 5, 4, 2, 7, 8, 3]], 5, 2),
        ([[6, 5, 4, 3]], [[6, 5, 4, 3, 7, 8, 9]], 5, 3),
    ],
)
def test_count_identity_switches(
    prev_frame_id, current_frame_id, n_gt, expected_output
):
    assert (
        count_identity_switches(prev_frame_id, current_frame_id, n_gt)
        == expected_output
    )


@pytest.mark.parametrize(
    "box1, box2, expected_iou",
    [
        ([0, 0, 10, 10], [5, 5, 12, 12], 0.25),
        ([0, 0, 10, 10], [0, 0, 10, 10], 1.0),
        ([0, 0, 10, 10], [20, 20, 30, 30], 0.0),
        ([0, 0, 10, 10], [5, 15, 15, 25], 0.0),
    ],
)
def test_calculate_iou(box1, box2, expected_iou):
    box1 = np.array(box1)
    box2 = np.array(box2)

    iou = calculate_iou(box1, box2)

    # Check if IoU matches expected value
    assert iou == pytest.approx(expected_iou, abs=1e-2)


@pytest.fixture
def gt_boxes():
    return np.array(
        [
            [10.0, 10.0, 20.0, 20.0, 1.0],
            [30.0, 30.0, 40.0, 40.0, 2.0],
            [50.0, 50.0, 60.0, 60.0, 3.0],
        ]
    )


@pytest.fixture
def tracked_boxes():
    return np.array(
        [
            [10.0, 10.0, 20.0, 20.0, 1.0],
            [30.0, 30.0, 40.0, 40.0, 2.0],
            [50.0, 50.0, 60.0, 60.0, 3.0],
        ]
    )


@pytest.fixture
def prev_frame_ids():
    return [[1.0, 2.0, 3.0]]


def test_perfect_tracking(gt_boxes, tracked_boxes, prev_frame_ids):
    mota = evaluate_mota(
        gt_boxes,
        tracked_boxes,
        iou_threshold=0.1,
        prev_frame_ids=prev_frame_ids,
    )
    assert mota == pytest.approx(1.0)


def test_missed_detections(gt_boxes, tracked_boxes, prev_frame_ids):
    # Remove one ground truth box to simulate a missed detection
    gt_boxes = np.delete(gt_boxes, 0, axis=0)
    mota = evaluate_mota(
        gt_boxes,
        tracked_boxes,
        iou_threshold=0.1,
        prev_frame_ids=prev_frame_ids,
    )
    assert mota < 1.0


def test_false_positives(gt_boxes, tracked_boxes, prev_frame_ids):
    # Add one extra tracked box to simulate a false positive
    tracked_boxes = np.vstack([tracked_boxes, [70, 70, 80, 80, 4]])
    mota = evaluate_mota(
        gt_boxes,
        tracked_boxes,
        iou_threshold=0.1,
        prev_frame_ids=prev_frame_ids,
    )
    assert mota < 1.0


def test_identity_switches(gt_boxes, tracked_boxes, prev_frame_ids):
    # Change ID of one tracked box to simulate an identity switch
    tracked_boxes[0][-1] = 5
    mota = evaluate_mota(
        gt_boxes,
        tracked_boxes,
        iou_threshold=0.5,
        prev_frame_ids=prev_frame_ids,
    )
    assert mota < 1.0


def test_get_ground_truth_data():
    test_csv_file = Path(__file__).parents[1] / "data" / "gt_test.csv"

    gt_data = get_ground_truth_data(test_csv_file)

    assert len(gt_data) == 2

    for i, frame_data in enumerate(gt_data):
        for j, detection_data in enumerate(frame_data):
            assert detection_data.shape == (
                5,
            ), f"Detection data shape mismatch for frame {i}"

    expected_ids = [2.0, 1.0]
    for i, frame_data in enumerate(gt_data):
        for j, detection_data in enumerate(frame_data):
            assert (
                detection_data[4] == expected_ids[j]
            ), f"Failed for frame {i}, detection {j}"


@pytest.fixture
def ground_truth_data():
    return [
        {
            "frame_number": 0,
            "x": 10,
            "y": 20,
            "width": 30,
            "height": 40,
            "id": 1,
        },
        {
            "frame_number": 0,
            "x": 50,
            "y": 60,
            "width": 70,
            "height": 80,
            "id": 2,
        },
        {
            "frame_number": 1,
            "x": 100,
            "y": 200,
            "width": 300,
            "height": 400,
            "id": 1,
        },
    ]


@pytest.fixture
def gt_boxes_list():
    return [np.array([]) for _ in range(2)]  # Two frames


def test_create_gt_list(ground_truth_data, gt_boxes_list):
    created_gt = create_gt_list(ground_truth_data, gt_boxes_list)

    assert isinstance(created_gt, list)

    for item in created_gt:
        assert isinstance(item, np.ndarray)

    assert len(created_gt) == len(gt_boxes_list)

    for i, array in enumerate(created_gt):
        for box in array:
            assert box.shape == (5,)

    i = 0
    for gt_created in created_gt:
        for frame_number in range(len(gt_created)):
            gt_data = ground_truth_data[i]
            gt_boxes = gt_created[frame_number]

            assert gt_boxes[0] == gt_data["x"]
            assert gt_boxes[1] == gt_data["y"]
            assert gt_boxes[2] == gt_data["x"] + gt_data["width"]
            assert gt_boxes[3] == gt_data["y"] + gt_data["height"]
            assert gt_boxes[4] == gt_data["id"]
            i += 1


def test_create_gt_list_invalid_data(ground_truth_data):
    invalid_data = ground_truth_data[:]

    del invalid_data[0]["x"]
    with pytest.raises(KeyError):
        create_gt_list(invalid_data, [np.array([]) for _ in range(2)])


def test_create_gt_list_insufficient_gt_boxes_list(ground_truth_data):
    with pytest.raises(IndexError):
        create_gt_list(ground_truth_data, [np.array([])])


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

    write_tracked_bbox_to_csv(bbox, frame, frame_name, csv_writer)

    expected_row = (
        "frame_0001.png",
        30000,
        '"{""clip"":123}"',
        1,
        0,
        '"{""name"":""rect"",""x"":10,""y"":20,""width"":40,""height"":60}"',
        '"{""track"":""1""}"',
    )
    expected_row_str = ",".join(map(str, expected_row))
    assert csv_output.getvalue().strip() == expected_row_str
