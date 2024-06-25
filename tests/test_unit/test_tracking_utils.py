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
    "prev_frame_id, current_frame_id, expected_output",
    [
        (None, {1: 1, 2: 2, 3: 3, 4: 4}, 0),
        ({1: 1, 2: 2, 3: 3, 4: 4}, {1: 1, 2: 2, 3: 3, 4: 4}, 0),
        ({1: 1, 2: 2, 3: 3, 4: 4}, {1: 2, 2: 1, 3: 3, 4: 4}, 2),
        ({1: 1, 2: 2, 3: 3, 4: 4}, {1: 1, 2: 2, 3: 3}, 0),
        ({1: 1, 2: 2, 3: 3}, {1: 1, 2: 2, 3: 3, 4: 4}, 0),
        ({1: 1, 2: 2, 3: 3, 4: 4}, {1: 1, 2: 2, 3: 3, 4: 5}, 1),
        ({1: 1, 2: 2, 3: 3, 4: 4}, {1: 1, 2: 2, 3: 3, 4: 5, 5: 6}, 2),
    ],
)
def test_count_identity_switches(
    prev_frame_id, current_frame_id, expected_output
):
    assert (
        count_identity_switches(prev_frame_id, current_frame_id)
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
def gt_ids():
    return [1, 2, 4]


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
def prev_frame_id_map():
    return {1: 1, 2: 2, 3: 3}


def test_perfect_tracking(gt_boxes, gt_ids, tracked_boxes, prev_frame_id_map):
    mota, _ = evaluate_mota(
        gt_boxes,
        gt_ids,
        tracked_boxes,
        iou_threshold=0.1,
        prev_frame_id_map=prev_frame_id_map,
    )
    assert mota == pytest.approx(1.0)
    # assert mota == 0


def test_missed_detections(gt_boxes, gt_ids, tracked_boxes, prev_frame_ids):
    # Remove one tracked box to simulate a missed detection
    tracked_boxes = np.delete(tracked_boxes, 0, axis=0)
    mota = evaluate_mota(
        gt_boxes,
        gt_ids,
        tracked_boxes,
        iou_threshold=0.1,
        prev_frame_ids=prev_frame_ids,
    )
    # mota = 1 - (1 + 0 + 0) / 3
    assert mota < 1.0
    assert mota == pytest.approx(2 / 3)


def test_false_positives(gt_boxes, gt_ids, tracked_boxes, prev_frame_ids):
    # Add one extra tracked box to simulate a false positive
    tracked_boxes = np.vstack([tracked_boxes, [70, 70, 80, 80, 4]])
    mota = evaluate_mota(
        gt_boxes,
        gt_ids,
        tracked_boxes,
        iou_threshold=0.1,
        prev_frame_ids=prev_frame_ids,
    )
    # mota = 1 - (0 + 1 + 0) / 3
    assert mota < 1.0
    assert mota == pytest.approx(2 / 3)


def test_identity_switches(gt_boxes, gt_ids, tracked_boxes, prev_frame_ids):
    # Change ID of one tracked box to simulate an identity switch
    tracked_boxes[0][-1] = 5
    mota = evaluate_mota(
        gt_boxes,
        gt_ids,
        tracked_boxes,
        iou_threshold=0.1,
        prev_frame_ids=prev_frame_ids,
    )
    # mota = 1 - (0 + 0 + 1) / 3
    assert mota < 1.0
    assert mota == pytest.approx(2 / 3)


def test_low_iou_false_positive_and_missed_detection(
    gt_boxes, gt_ids, tracked_boxes, prev_frame_ids
):
    # set one of the tracked_box to have low IOU
    tracked_boxes[1] = [30.0, 30.0, 30.0, 30.0, 2.0]
    mota = evaluate_mota(
        gt_boxes,
        gt_ids,
        tracked_boxes,
        iou_threshold=0.1,
        prev_frame_ids=prev_frame_ids,
    )
    # mota = 1 - (1 + 1 + 0) / 3
    assert mota < 1.0
    assert mota == pytest.approx(1 / 3)


def test_mota_zero(gt_boxes, tracked_boxes, prev_frame_ids):
    # set one of the tracked_box to have low IOU and one of tracked_box has ID switch
    tracked_boxes[1] = [30.0, 30.0, 30.0, 30.0, 2.0]
    tracked_boxes[0][-1] = 4
    mota = evaluate_mota(
        gt_boxes,
        gt_ids,
        tracked_boxes,
        iou_threshold=0.1,
        prev_frame_ids=prev_frame_ids,
    )
    # mota = 1 - (1 + 1 + 1) / 3
    assert mota == 0


def test_more_than_one_switches(
    gt_boxes, gt_ids, tracked_boxes, prev_frame_ids
):
    # set one of the tracked_box to have low IOU and the other two tracked_boxes have ID switch
    tracked_boxes[1] = [30.0, 30.0, 30.0, 30.0, 2.0]
    tracked_boxes[0][-1] = 4
    tracked_boxes[2][-1] = 5
    mota = evaluate_mota(
        gt_boxes,
        gt_ids,
        tracked_boxes,
        iou_threshold=0.1,
        prev_frame_ids=prev_frame_ids,
    )
    # mota = 1 - (1 + 1 + 2) / 3
    assert mota < 0
    assert mota == pytest.approx(-1 / 3)


def test_more_than_one_switches_same_box(
    gt_boxes, gt_ids, tracked_boxes, prev_frame_ids
):
    # set one of the tracked_box to have low IOU and
    # the two tracked_boxes have ID switch (one is the one with low IOU)
    tracked_boxes[1] = [30.0, 30.0, 30.0, 30.0, 2.0]
    tracked_boxes[1][-1] = 4
    tracked_boxes[2][-1] = 5
    mota = evaluate_mota(
        gt_boxes,
        gt_ids,
        tracked_boxes,
        iou_threshold=0.1,
        prev_frame_ids=prev_frame_ids,
    )
    # mota = 1 - (1 + 1 + 1) / 3
    assert mota == 0


def test_get_ground_truth_data():
    test_csv_file = Path(__file__).parents[1] / "data" / "gt_test.csv"

    gt_boxes_list, gt_ids_list = get_ground_truth_data(test_csv_file)

    assert isinstance(gt_boxes_list, list)
    assert isinstance(gt_ids_list, list)
    assert all(isinstance(arr, np.ndarray) for arr in gt_boxes_list)
    assert all(isinstance(arr, np.ndarray) for arr in gt_ids_list)

    for frame_data in gt_boxes_list:
        if frame_data.size > 0:
            assert frame_data.shape[1] == 4, "Bounding box shape mismatch"


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


@pytest.fixture
def gt_ids_list():
    return [np.array([]) for _ in range(2)]  # Two frames


def test_create_gt_list(ground_truth_data, gt_boxes_list, gt_ids_list):
    created_gt_boxes, created_gt_ids = create_gt_list(
        ground_truth_data, gt_boxes_list, gt_ids_list
    )

    assert isinstance(created_gt_boxes, list)
    assert isinstance(created_gt_ids, list)

    for item in created_gt_boxes:
        assert isinstance(item, np.ndarray)

    for item in created_gt_ids:
        assert isinstance(item, np.ndarray)

    assert len(created_gt_boxes) == len(gt_boxes_list)
    assert len(created_gt_ids) == len(gt_ids_list)

    # Check the bounding box and ID data
    expected_boxes = [
        np.array([[10, 20, 40, 60], [50, 60, 120, 140]]),
        np.array([[100, 200, 400, 600]]),
    ]
    expected_ids = [
        np.array([1, 2]),
        np.array([1]),
    ]

    for i, (boxes, ids) in enumerate(zip(created_gt_boxes, created_gt_ids)):
        if boxes.size > 0:
            assert np.array_equal(boxes, expected_boxes[i])
        if ids.size > 0:
            assert np.array_equal(ids, expected_ids[i])


def test_create_gt_list_invalid_data(
    ground_truth_data, gt_boxes_list, gt_ids_list
):
    invalid_data = ground_truth_data[:]

    del invalid_data[0]["x"]
    with pytest.raises(KeyError):
        create_gt_list(invalid_data, gt_boxes_list, gt_ids_list)


def test_create_gt_list_insufficient_gt_boxes_list(ground_truth_data):
    with pytest.raises(IndexError):
        create_gt_list(ground_truth_data, [np.array([])], [np.array([])])


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
