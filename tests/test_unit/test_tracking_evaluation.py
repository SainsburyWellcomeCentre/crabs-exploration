from pathlib import Path

import numpy as np
import pytest

from crabs.tracker.evaluate_tracker import TrackerEvaluate


@pytest.fixture
def evaluation():
    test_csv_file = Path(__file__).parents[1] / "data" / "gt_test.csv"
    return TrackerEvaluate(test_csv_file, tracked_list=[], iou_threshold=0.1)


def test_get_ground_truth_data(evaluation):
    gt_boxes_list, gt_ids_list = evaluation.get_ground_truth_data()

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


def test_create_gt_list(
    ground_truth_data, gt_boxes_list, gt_ids_list, evaluation
):
    created_gt_boxes, created_gt_ids = evaluation.create_gt_list(
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
    ground_truth_data, gt_ids_list, evaluation
):
    invalid_data = ground_truth_data[:]

    del invalid_data[0]["x"]
    with pytest.raises(KeyError):
        evaluation.create_gt_list(
            invalid_data, [np.array([]) for _ in range(2)], gt_ids_list
        )


def test_create_gt_list_insufficient_gt_boxes_list(
    ground_truth_data, gt_ids_list, evaluation
):
    with pytest.raises(IndexError):
        evaluation.create_gt_list(
            ground_truth_data, [np.array([])], gt_ids_list
        )


@pytest.mark.parametrize(
    "prev_frame_id_map, current_frame_id_map, expected_output",
    [
        (None, {1: 11, 2: 12, 3: 13, 4: 14}, 0),
        ({1: 11, 2: 12, 3: 13, 4: 14}, {1: 11, 2: 12, 3: 13, 4: 14}, 0),
        ({1: 11, 2: 12, 3: 13, 4: 14}, {1: 12, 2: 11, 3: 13, 4: 14}, 2),
        ({1: 11, 2: 12, 3: 13, 4: 14}, {1: 11, 2: 12, 3: 13}, 0),
        ({1: 11, 2: 12, 3: 13}, {1: 11, 2: 12, 3: 13, 4: 14}, 0),
        ({1: 11, 2: 12, 3: 13}, {1: 11, 2: 12, 4: 13}, 1),
        ({1: 11, 2: 12, 3: 13, 4: 14}, {1: 11, 2: 12, 3: 13, 4: 15}, 1),
        ({1: 11, 2: 12, 3: 13, 4: 14}, {1: 11, 2: 12, 3: 13, 4: 15, 5: 16}, 1),
        ({3: 23, 4: 100, 1: 11, 2: 21}, {4: 23, 3: 100, 1: 11, 2: 21}, 2),
    ],
)
def test_count_identity_switches(
    evaluation, prev_frame_id_map, current_frame_id_map, expected_output
):
    assert (
        evaluation.count_identity_switches(
            prev_frame_id_map, current_frame_id_map
        )
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


@pytest.mark.parametrize(
    "gt_boxes, gt_ids, tracked_boxes, prev_frame_id_map, expected_mota",
    [
        # perfect tracking
        (
            np.array(
                [
                    [10.0, 10.0, 20.0, 20.0],
                    [30.0, 30.0, 40.0, 40.0],
                    [50.0, 50.0, 60.0, 60.0],
                ]
            ),
            [1, 2, 3],
            np.array(
                [
                    [10.0, 10.0, 20.0, 20.0, 11],
                    [30.0, 30.0, 40.0, 40.0, 12],
                    [50.0, 50.0, 60.0, 60.0, 13],
                ]
            ),
            {1: 11, 12: 2, 3: 13},
            1.0,
        ),
        # prev_map = {1: 11, 2: 12, 3: 13}, current_map = {1: 11, 2: 12, 4: 14}
        (
            np.array(
                [
                    [10.0, 10.0, 20.0, 20.0],
                    [30.0, 30.0, 40.0, 40.0],
                    [50.0, 50.0, 60.0, 60.0],
                ]
            ),
            [1, 2, 4],
            np.array(
                [
                    [10.0, 10.0, 20.0, 20.0, 11],
                    [30.0, 30.0, 40.0, 40.0, 12],
                    [50.0, 50.0, 60.0, 60.0, 14],
                ]
            ),
            {1: 11, 2: 12, 3: 13},
            1.0,
        ),
        # missed detection
        (
            np.array(
                [
                    [10.0, 10.0, 20.0, 20.0],
                    [30.0, 30.0, 40.0, 40.0],
                    [50.0, 50.0, 60.0, 60.0],
                ]
            ),
            [1, 2, 4],
            np.array(
                [
                    [10.0, 10.0, 20.0, 20.0, 11],
                    [30.0, 30.0, 40.0, 40.0, 12],
                ]
            ),
            {1: 11, 2: 12, 3: 13},
            2 / 3,
        ),
        # false positive
        (
            np.array(
                [
                    [10.0, 10.0, 20.0, 20.0],
                    [30.0, 30.0, 40.0, 40.0],
                    [50.0, 50.0, 60.0, 60.0],
                ]
            ),
            [1, 2, 3],
            np.array(
                [
                    [10.0, 10.0, 20.0, 20.0, 11],
                    [30.0, 30.0, 40.0, 40.0, 12],
                    [50.0, 50.0, 60.0, 60.0, 13],
                    [70.0, 70.0, 80.0, 80.0, 14],
                ]
            ),
            {1: 11, 2: 12, 3: 13},
            2 / 3,
        ),
        # one with low IOU and another one has ID switch
        (
            np.array(
                [
                    [10.0, 10.0, 20.0, 20.0],
                    [30.0, 30.0, 40.0, 40.0],
                    [50.0, 50.0, 60.0, 60.0],
                ]
            ),
            [1, 2, 3],
            np.array(
                [
                    [10.0, 10.0, 20.0, 20.0, 11],
                    [30.0, 30.0, 30.0, 30.0, 12],
                    [50.0, 50.0, 60.0, 60.0, 14],
                ]
            ),
            {1: 11, 2: 12, 3: 13},
            0,
        ),
        # low IOU and one ID switch on the same box
        (
            np.array(
                [
                    [10.0, 10.0, 20.0, 20.0],
                    [30.0, 30.0, 40.0, 40.0],
                    [50.0, 50.0, 60.0, 60.0],
                ]
            ),
            [1, 2, 3],
            np.array(
                [
                    [10.0, 10.0, 20.0, 20.0, 11],
                    [30.0, 30.0, 30.0, 30.0, 14],
                    [50.0, 50.0, 60.0, 60.0, 13],
                ]
            ),
            {1: 11, 2: 12, 3: 13},
            1 / 3,
        ),
        # current tracked id = prev tracked id, but prev_gt_id != current gt id
        (
            np.array(
                [
                    [10.0, 10.0, 20.0, 20.0],
                    [30.0, 30.0, 40.0, 40.0],
                    [50.0, 50.0, 60.0, 60.0],
                ]
            ),
            [1, 2, 4],
            np.array(
                [
                    [10.0, 10.0, 20.0, 20.0, 11],
                    [30.0, 30.0, 40.0, 40.0, 12],
                    [50.0, 50.0, 60.0, 60.0, 13],
                ]
            ),
            {1: 11, 2: 12, 3: 13},
            2 / 3,
        ),
        # ID swapped
        (
            np.array(
                [
                    [10.0, 10.0, 20.0, 20.0],
                    [30.0, 30.0, 40.0, 40.0],
                    [50.0, 50.0, 60.0, 60.0],
                ]
            ),
            [1, 2, 3],
            np.array(
                [
                    [10.0, 10.0, 20.0, 20.0, 11],
                    [30.0, 30.0, 40.0, 40.0, 13],
                    [50.0, 50.0, 60.0, 60.0, 12],
                ]
            ),
            {1: 11, 2: 12, 3: 13},
            1 / 3,
        ),
    ],
)
def test_evaluate_mota(
    gt_boxes,
    gt_ids,
    tracked_boxes,
    prev_frame_id_map,
    expected_mota,
    evaluation,
):
    mota, _ = evaluation.evaluate_mota(
        gt_boxes,
        gt_ids,
        tracked_boxes,
        iou_threshold=0.1,
        prev_frame_id_map=prev_frame_id_map,
    )
    assert mota == pytest.approx(expected_mota)
