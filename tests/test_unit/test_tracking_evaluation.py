from pathlib import Path

import numpy as np
import pytest

from crabs.tracker.evaluate_tracker import TrackerEvaluate


@pytest.fixture
def evaluation():
    test_csv_file = Path(__file__).parents[1] / "data" / "gt_test.csv"
    return TrackerEvaluate(
        test_csv_file,
        tracked_list=[],
        iou_threshold=0.1,
        video_name="test.mp4",
        ckpt_name="/checkpoint/last.ckpt",
    )


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


def test_get_ground_truth_data(evaluation):
    gt_data = evaluation.get_ground_truth_data()

    assert len(gt_data) == 2

    gt_boxes_list, gt_ids_list = gt_data

    for frame_data in gt_boxes_list:
        assert isinstance(frame_data, np.ndarray)

    for frame_data in gt_ids_list:
        assert isinstance(frame_data, np.ndarray)


def test_create_gt_list(
    ground_truth_data, gt_boxes_list, gt_ids_list, evaluation
):
    created_gt_boxes_list, created_gt_ids_list = evaluation.create_gt_list(
        ground_truth_data, gt_boxes_list, gt_ids_list
    )

    assert isinstance(created_gt_boxes_list, list)
    assert isinstance(created_gt_ids_list, list)

    assert len(created_gt_boxes_list) == len(gt_boxes_list)
    assert len(created_gt_ids_list) == len(gt_ids_list)

    for i, frame_data in enumerate(created_gt_boxes_list):
        for detection_data in frame_data:
            assert detection_data.shape == (
                4,
            ), f"Detection data shape mismatch for frame {i}"

    expected_boxes = [
        np.array([[10, 20, 40, 60], [50, 60, 120, 140]], dtype=np.float32),
        np.array([[100, 200, 400, 600]], dtype=np.float32),
    ]
    expected_ids = [
        np.array([1, 2], dtype=np.float32),
        np.array([1], dtype=np.float32),
    ]

    for i, (boxes, ids) in enumerate(
        zip(created_gt_boxes_list, created_gt_ids_list)
    ):
        assert np.array_equal(
            boxes, expected_boxes[i]
        ), f"Mismatch boxes for frame {i}"
        assert np.array_equal(
            ids, expected_ids[i]
        ), f"Mismatch ID for frame {i}"


def test_create_gt_list_invalid_data(
    ground_truth_data, gt_boxes_list, gt_ids_list, evaluation
):
    invalid_data = ground_truth_data[:]
    invalid_data[0].pop("x")  # Remove a required key to simulate invalid data

    with pytest.raises(KeyError):
        evaluation.create_gt_list(invalid_data, gt_boxes_list, gt_ids_list)


def test_create_gt_list_insufficient_gt_boxes_list(
    ground_truth_data, evaluation
):
    with pytest.raises(IndexError):
        evaluation.create_gt_list(
            ground_truth_data, [np.array([])], [np.array([])]
        )


@pytest.mark.parametrize(
    "prev_frame_id_map, current_frame_id_map, expected_output",
    [
        (None, {1: 11, 2: 12, 3: 13, 4: 14}, 0),
        ({1: 11, 2: 12, 3: 13, 4: 14}, {1: 11, 2: 12, 3: 13, 4: 14}, 0),
        ({1: 11, 2: 12, 3: 13, 4: 14}, {1: 12, 2: 11, 3: 13, 4: 14}, 2),
        ({1: 11, 2: 12, 3: 13, 4: 14}, {1: 11, 2: 12, 3: 13}, 0),
        ({1: 11, 2: 12, 3: 13, 4: 14}, {1: 11, 2: 12, 3: 13, 5: 14}, 1),
        ({1: 11, 2: 12, 3: 13}, {1: 11, 2: 12, 3: 13, 4: 14}, 0),
        ({1: 11, 2: 12, 3: 13}, {1: 11, 2: 12, 4: 13}, 1),
        ({1: 11, 2: 12, 3: 13, 4: 14}, {1: 11, 2: 12, 3: 13, 4: 15}, 1),
        ({1: 11, 2: 12, 3: 13, 4: 14}, {1: 11, 2: 12, 3: 13, 4: 15, 5: 16}, 1),
        ({3: 23, 4: 100, 1: 11, 2: 21}, {4: 23, 3: 100, 1: 11, 2: 21}, 2),
        ({3: 28, 1: 11, 2: 34}, {4: 28, 1: 11, 2: 34}, 1),
        ({1: 11, 2: 12, 3: 13}, {2: 12, 3: 14, 4: 13}, 2),
        ({1: 11, 2: 12, 3: 13}, {1: 11, 2: 14, 3: 13}, 1),
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


@pytest.fixture
def gt_boxes():
    return np.array(
        [
            [10.0, 10.0, 20.0, 20.0],
            [30.0, 30.0, 40.0, 40.0],
            [50.0, 50.0, 60.0, 60.0],
        ]
    )


@pytest.fixture
def gt_ids():
    return [1, 2, 3]


@pytest.fixture
def tracked_boxes():
    return np.array(
        [
            [10.0, 10.0, 20.0, 20.0, 11],
            [30.0, 30.0, 40.0, 40.0, 12],
            [50.0, 50.0, 60.0, 60.0, 13],
        ]
    )


@pytest.fixture
def prev_frame_id_map():
    return {1: 11, 2: 12, 3: 13}


def test_perfect_tracking(
    gt_boxes, gt_ids, tracked_boxes, prev_frame_id_map, evaluation
):
    mota, true_positive, _, _, _, _, _ = evaluation.evaluate_mota(
        gt_boxes,
        gt_ids,
        tracked_boxes,
        prev_frame_id_map,
    )
    assert mota == pytest.approx(1.0)
    assert true_positive == len(gt_boxes)


def test_missed_detections(
    gt_boxes, gt_ids, tracked_boxes, prev_frame_id_map, evaluation
):
    # Remove one tracked box to simulate a missed detection
    tracked_boxes = np.delete(tracked_boxes, 0, axis=0)
    mota, _, missed_detection, _, _, _, _ = evaluation.evaluate_mota(
        gt_boxes,
        gt_ids,
        tracked_boxes,
        prev_frame_id_map,
    )
    assert mota < 1.0
    assert missed_detection == 1


def test_false_positives(
    gt_boxes, gt_ids, tracked_boxes, prev_frame_id_map, evaluation
):
    # Add one extra tracked box to simulate a false positive
    tracked_boxes = np.vstack([tracked_boxes, [70, 70, 80, 80, 14]])
    mota, _, _, false_positive, _, _, _ = evaluation.evaluate_mota(
        gt_boxes,
        gt_ids,
        tracked_boxes,
        prev_frame_id_map,
    )
    assert mota < 1.0
    assert false_positive == 1


def test_new_id_no_switch(
    gt_boxes, gt_ids, tracked_boxes, prev_frame_id_map, evaluation
):
    # prev_map = {1: 11, 2: 12, 3: 13}, current_map = {1: 11, 2: 12, 4: 14}
    gt_ids[2] = 4
    tracked_boxes[2][-1] = 14
    mota, true_positive, _, _, num_switches, _, _ = evaluation.evaluate_mota(
        gt_boxes,
        gt_ids,
        tracked_boxes,
        prev_frame_id_map,
    )
    assert mota == 1.0
    assert true_positive == len(gt_boxes)
    assert num_switches == 0


def test_low_iou_and_switch(
    gt_boxes, gt_ids, tracked_boxes, prev_frame_id_map, evaluation
):
    # one with low IOU and another one has ID switch
    tracked_boxes[1] = [30.0, 30.0, 30.0, 30.0, 12]
    tracked_boxes[2][-1] = 14
    (
        mota,
        _,
        missed_detections,
        false_positive,
        num_switches,
        _,
        _,
    ) = evaluation.evaluate_mota(
        gt_boxes,
        gt_ids,
        tracked_boxes,
        prev_frame_id_map,
    )
    assert mota == 0.0
    assert missed_detections == 1
    assert false_positive == 1
    assert num_switches == 1


def test_low_iou_with_switch(
    gt_boxes, gt_ids, tracked_boxes, prev_frame_id_map, evaluation
):
    # one with low IOU and another one has ID switch
    tracked_boxes[1] = [30.0, 30.0, 30.0, 30.0, 14]
    (
        mota,
        _,
        missed_detections,
        false_positive,
        num_switches,
        _,
        _,
    ) = evaluation.evaluate_mota(
        gt_boxes,
        gt_ids,
        tracked_boxes,
        prev_frame_id_map,
    )
    assert mota == pytest.approx(1 / 3)
    assert missed_detections == 1
    assert false_positive == 1
    assert num_switches == 0


def test_id_swapped(
    gt_boxes, gt_ids, tracked_boxes, prev_frame_id_map, evaluation
):
    # one with low IOU and another one has ID switch
    tracked_boxes[1][-1] = 13
    tracked_boxes[2][-1] = 12
    mota, _, _, _, num_switches, _, _ = evaluation.evaluate_mota(
        gt_boxes,
        gt_ids,
        tracked_boxes,
        prev_frame_id_map,
    )
    assert mota == pytest.approx(1 / 3)
    assert num_switches == 2


# @pytest.fixture
# def sample_csv_data():
#     # Create a sample CSV file with some data
#     sample_data = [
#         {
#             "True Positives": "10",
#             "Missed Detections": "2",
#             "False Positives": "3",
#             "Number of Switches": "1",
#             "Total Ground Truth": "15",
#             "Mota": "0.8",
#         },
#         {
#             "True Positives": "15",
#             "Missed Detections": "3",
#             "False Positives": "2",
#             "Number of Switches": "2",
#             "Total Ground Truth": "20",
#             "Mota": "0.9",
#         },
#     ]
#     with tempfile.NamedTemporaryFile(mode="w", delete=False) as temp_file:
#         writer = csv.DictWriter(temp_file, fieldnames=sample_data[0].keys())
#         writer.writeheader()
#         writer.writerows(sample_data)
#         temp_file_path = temp_file.name
#     yield temp_file_path
#     # Clean up after the test
#     import os

#     os.remove(temp_file_path)


# def test_read_metrics_from_csv(sample_csv_data):
#     (
#         true_positives_list,
#         missed_detections_list,
#         false_positives_list,
#         num_switches_list,
#         total_ground_truth_list,
#         mota_value_list,
#     ) = read_metrics_from_csv(sample_csv_data)

#     assert true_positives_list == [10, 15]
#     assert missed_detections_list == [2, 3]
#     assert false_positives_list == [3, 2]
#     assert num_switches_list == [1, 2]
#     assert total_ground_truth_list == [15, 20]
#     assert mota_value_list == [0.8, 0.9]
