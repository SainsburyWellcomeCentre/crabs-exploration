from pathlib import Path

import numpy as np
import pytest

from crabs.tracking.evaluation import Evaluation


@pytest.fixture
def evaluation():
    test_csv_file = Path(__file__).parents[1] / "data" / "gt_test.csv"
    return Evaluation(test_csv_file, tracked_list=[], iou_threshold=0.1)


def test_get_ground_truth_data(evaluation):
    gt_data = evaluation.get_ground_truth_data()

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


def test_create_gt_list(ground_truth_data, gt_boxes_list, evaluation):
    created_gt = evaluation.create_gt_list(ground_truth_data, gt_boxes_list)

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


def test_create_gt_list_invalid_data(ground_truth_data, evaluation):
    invalid_data = ground_truth_data[:]

    del invalid_data[0]["x"]
    with pytest.raises(KeyError):
        evaluation.create_gt_list(
            invalid_data, [np.array([]) for _ in range(2)]
        )


def test_create_gt_list_insufficient_gt_boxes_list(
    ground_truth_data, evaluation
):
    with pytest.raises(IndexError):
        evaluation.create_gt_list(ground_truth_data, [np.array([])])


@pytest.mark.parametrize(
    "prev_frame_id, current_frame_id, expected_output",
    [
        (None, [[6, 5, 4, 3, 2, 1]], 0),
        (
            [[6, 5, 4, 3, 2, 1]],
            [[6, 5, 4, 3, 2, 1]],
            0,
        ),  # no identity switches
        ([[5, 6, 4, 3, 1, 2]], [[6, 5, 4, 3, 2, 1]], 0),
        ([[6, 5, 4, 3, 2, 1]], [[6, 5, 4, 2, 1]], 1),
        ([[6, 5, 4, 2, 1]], [[6, 5, 4, 2, 1, 7]], 1),
        ([[6, 5, 4, 2, 1, 7]], [[6, 5, 4, 2, 7, 8]], 2),
        ([[6, 5, 4, 2, 7, 8]], [[6, 5, 4, 2, 7, 8, 3]], 1),
    ],
)
def test_count_identity_switches(
    evaluation, prev_frame_id, current_frame_id, expected_output
):
    assert (
        evaluation.count_identity_switches(prev_frame_id, current_frame_id)
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
def test_calculate_iou(box1, box2, expected_iou, evaluation):
    box1 = np.array(box1)
    box2 = np.array(box2)

    iou = evaluation.calculate_iou(box1, box2)

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


def test_perfect_tracking(gt_boxes, tracked_boxes, prev_frame_ids, evaluation):
    mota = evaluation.evaluate_mota(
        gt_boxes,
        tracked_boxes,
        iou_threshold=0.1,
        prev_frame_ids=prev_frame_ids,
    )
    assert mota == pytest.approx(1.0)


def test_missed_detections(
    gt_boxes, tracked_boxes, prev_frame_ids, evaluation
):
    # Remove one ground truth box to simulate a missed detection
    gt_boxes = np.delete(gt_boxes, 0, axis=0)
    mota = evaluation.evaluate_mota(
        gt_boxes,
        tracked_boxes,
        iou_threshold=0.1,
        prev_frame_ids=prev_frame_ids,
    )
    assert mota < 1.0


def test_false_positives(gt_boxes, tracked_boxes, prev_frame_ids, evaluation):
    # Add one extra tracked box to simulate a false positive
    tracked_boxes = np.vstack([tracked_boxes, [70, 70, 80, 80, 4]])
    mota = evaluation.evaluate_mota(
        gt_boxes,
        tracked_boxes,
        iou_threshold=0.1,
        prev_frame_ids=prev_frame_ids,
    )
    assert mota < 1.0


def test_identity_switches(
    gt_boxes, tracked_boxes, prev_frame_ids, evaluation
):
    # Change ID of one tracked box to simulate an identity switch
    tracked_boxes[0][-1] = 5
    mota = evaluation.evaluate_mota(
        gt_boxes,
        tracked_boxes,
        iou_threshold=0.5,
        prev_frame_ids=prev_frame_ids,
    )
    assert mota < 1.0
