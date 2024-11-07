from pathlib import Path

import numpy as np
import pytest

from crabs.tracker.evaluate_tracker import TrackerEvaluate


@pytest.fixture
def tracker_evaluate_interface():
    annotations_file_csv = Path(__file__).parents[1] / "data" / "gt_test.csv"
    return TrackerEvaluate(
        annotations_file_csv,
        predicted_boxes_dict={},
        iou_threshold=0.1,
        tracking_output_dir="/path/output",
    )


def test_get_ground_truth_data_structure(tracker_evaluate_interface):
    """Test the loaded ground truth data has the expected structure."""
    # Get ground truth data dict
    ground_truth_dict = tracker_evaluate_interface.get_ground_truth_data()

    # check type
    assert isinstance(ground_truth_dict, dict)
    # check it is a nested dictionary
    assert all(
        isinstance(frame_data, dict)
        for frame_data in ground_truth_dict.values()
    )

    # check data types for values in nested dictionary
    for frame_number, data in ground_truth_dict.items():
        assert isinstance(frame_number, int)
        assert isinstance(data["bbox"], np.ndarray)
        assert isinstance(data["id"], np.ndarray)
        assert data["bbox"].shape[1] == 4


def test_ground_truth_data_values(tracker_evaluate_interface):
    """Test ground truth data holds expected values."""
    # Define expected ground truth data
    expected_data = {
        11: {
            "bbox": np.array(
                [
                    [2894.8606, 975.8517, 2945.8606, 1016.8517],
                    [940.6089, 1192.637, 989.6089, 1230.637],
                ],
                dtype=np.float32,
            ),
            "id": np.array([2.0, 1.0], dtype=np.float32),
        },
        21: {
            "bbox": np.array(
                [[940.6089, 1192.637, 989.6089, 1230.637]], dtype=np.float32
            ),
            "id": np.array([2.0], dtype=np.float32),
        },
    }

    # Get ground truth data dict
    ground_truth_dict = tracker_evaluate_interface.get_ground_truth_data()

    # Check if ground truth data matches expected values
    for expected_frame_number, expected_frame_data in expected_data.items():
        # check expected key is present
        assert expected_frame_number in ground_truth_dict

        # check n of bounding boxes per frame matches the expected value
        assert len(ground_truth_dict[expected_frame_number]["bbox"]) == len(
            expected_frame_data["bbox"]
        )

        # check bbox arrays match the expected values
        for bbox, expected_bbox in zip(
            ground_truth_dict[expected_frame_number]["bbox"],
            expected_frame_data["bbox"],
        ):
            assert np.allclose(
                bbox, expected_bbox
            ), f"Frame {expected_frame_number}, bbox mismatch"

        # check id arrays match the expected values
        assert np.array_equal(
            ground_truth_dict[expected_frame_number]["id"],
            expected_frame_data["id"],
        ), f"Frame {expected_frame_number}, id mismatch"


@pytest.mark.parametrize(
    "prev_frame_id_map, current_frame_id_map, expected_output",
    [
        (None, {1: 11, 2: 12, 3: 13, 4: 14}, 0),  # no previous frame
        # ----- a crab (GT=3) that continues to exist ---------
        (
            {1: 11, 2: 12, 3: 13, 4: 14},
            {1: 11, 2: 12, 3: 13, 4: 14},
            0,
        ),  # correct
        (
            {1: 11, 2: 12, 3: 13, 4: 14},
            {1: 11, 2: 12, 3: np.nan, 4: 14},
            0,
        ),  # crab is missed detection in current frame
        (
            {1: 11, 2: 12, 3: np.nan, 4: 14},
            {1: 11, 2: 12, 3: 13, 4: 14},
            0,
        ),  # crab is missed detection in previous frame
        (
            {1: 11, 2: 12, 3: 13, 4: 14},
            {1: 11, 2: 12, 3: 15, 4: 14},
            1,
        ),  # crab is re-IDed in current frame
        (
            {1: 11, 2: 12, 3: 13, 4: 14},
            {1: 11, 2: 12, 3: 14},
            1,
        ),  # crab swaps ID with a disappearing crab
        (
            {1: 11, 2: 12, 3: 13},
            {1: 11, 2: 12, 4: 13},
            1,
        ),  # disappear crab swaps ID with an appearing crab
        (
            {1: 11, 2: 12, 3: 13},
            {1: 11, 2: 12, 3: 99, 4: 13},
            2,
        ),  # crab swaps ID with an appearing crab
        (
            {1: 11, 2: 12, 3: 13, 4: 14},
            {1: 11, 2: 12, 3: 14, 4: 13},
            2,
        ),  # crab swaps ID with another crab that continues to exist
        # ----- a crab (GT=4) disappears ---------
        (
            {1: 11, 2: 12, 3: 13, 4: 14},
            {1: 11, 2: 12, 3: 13},
            0,
        ),  # correct
        (
            {1: 11, 2: 12, 3: 13, 4: 14},
            {1: 11, 2: 12, 3: 14},
            1,
        ),  # crab disappears and another pre-existing one takes its ID
        (
            {1: 11, 2: 12, 3: 13, 4: 14},
            {1: 11, 2: 12, 3: 13, 5: 14},
            1,
        ),  # crab disappears and an appearing one takes its ID
        (
            {1: 11, 2: 12, 3: 13, 4: np.nan},
            {1: 11, 2: 12, 3: 13},
            0,
        ),  # crab disappears but was missed detection in frame f-1
        (
            {1: 11, 2: 12, 3: 13, 4: np.nan},
            {1: 11, 2: 12, 3: 13, 5: np.nan},
            0,
        ),  # crab disappears but was missed detection in frame f-1,
        # with a new missed crab in frame f
        (
            {1: 11, 2: 12, 3: 13, 4: np.nan},
            {1: 11, 2: 12, 3: np.nan},
            0,
        ),  # crab disappears but was missed detection in frame f-1,
        # and existing crab was missed in frame f
        # ----- a crab (GT=4) appears ---------
        (
            {1: 11, 2: 12, 3: 13},
            {1: 11, 2: 12, 3: 13, 4: 14},
            0,
        ),  # correct
        (
            {1: 11, 2: 12, 3: 14},
            {1: 11, 2: 12, 3: 13, 4: 14},
            2,
        ),  # crab that appears gets ID of a pre-existing crab
        (
            {1: 11, 2: 12, 3: 13},
            {1: 11, 2: 12, 4: 13},
            1,
        ),  # crab that appears gets ID of a crab that disappears
        (
            {1: 11, 2: 12, 3: 13},
            {1: 11, 2: 12, 3: 13, 4: np.nan},
            0,
        ),  # missed detection in current frame
        (
            {1: 11, 2: 12, 3: 13, 5: np.nan},
            {1: 11, 2: 12, 3: 13, 4: np.nan},
            0,
        ),  # crab that appears is missed detection in current frame,
        # and another missed detection in previous frame disappears
        (
            {1: 11, 2: 12, 3: np.nan},
            {1: 11, 2: 12, 3: 13, 4: np.nan},
            0,
        ),  # crab that appears is missed detection in current frame,
        # and a pre-existing crab is missed detection in previous frame
        # ----------
        # Test consistency with last predicted ID if a crab (GT=3)
        # that continues to exist is not detected for a few frames (>= 1)
        # ------------
        (
            {1: 11, 2: 12, 3: np.nan},
            {1: 11, 2: 12, 3: 13},
            0,
        ),  # crab that continues to exist, and the current predicted ID is
        # consistent with last_known_predicted_ids={1: 11, 2: 12, 3: 13, 4: 14}
        (
            {1: 11, 2: 12, 3: np.nan},
            {1: 11, 2: 12, 3: 14},
            1,
        ),  # crab that continues to exist, and the current predicted ID
        # is NOT consistent with the
        # last_known_predicted_ids={1: 11, 2: 12, 3: 13, 4: 14}
        # ----------
        # Test consistency with last predicted ID if a crab (GT=3)
        # re-appears after a few frames (>= 1)
        # ------------
        (
            {1: 11, 2: 12},
            {1: 11, 2: 12, 3: 13},
            0,
        ),  # crab whose GT ID is in last_known_predicted_ids, appears
        # in the current frame, and the current predicted ID is consistent
        # with last_known_predicted_ids
        (
            {1: 11, 2: 12},
            {1: 11, 2: 12, 3: 14},
            1,
        ),  # crab whose GT ID is in last_known_predicted_ids, appears
        # in the current frame, and the current predicted ID is NOT consistent
        # with last_known_predicted_ids
    ],
)
def test_count_identity_switches(
    tracker_evaluate_interface,
    prev_frame_id_map,
    current_frame_id_map,
    expected_output,
):
    tracker_evaluate_interface.last_known_predicted_ids = {
        1: 11,
        2: 12,
        3: 13,
        4: 14,
    }
    assert (
        tracker_evaluate_interface.count_identity_switches(
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
def test_calculate_iou(box1, box2, expected_iou, tracker_evaluate_interface):
    box1 = np.array(box1)
    box2 = np.array(box2)

    iou = tracker_evaluate_interface.calculate_iou(box1, box2)

    # Check if IoU matches expected value
    assert iou == pytest.approx(expected_iou, abs=1e-2)


@pytest.mark.parametrize(
    "gt_data, pred_data, prev_frame_id_map, expected_output",
    [
        # perfect tracking
        (
            {
                "bbox": np.array(
                    [
                        [10.0, 10.0, 20.0, 20.0],
                        [30.0, 30.0, 40.0, 40.0],
                        [50.0, 50.0, 60.0, 60.0],
                    ]
                ),
                "id": np.array([1, 2, 3]),
            },
            {
                "bbox": np.array(
                    [
                        [10.0, 10.0, 20.0, 20.0],
                        [30.0, 30.0, 40.0, 40.0],
                        [50.0, 50.0, 60.0, 60.0],
                    ]
                ),
                "id": np.array([11, 12, 13]),
            },
            {1: 11, 2: 12, 3: 13},
            [1.0, 3, 0, 0, 0],
        ),
        (
            {
                "bbox": np.array(
                    [
                        [10.0, 10.0, 20.0, 20.0],
                        [30.0, 30.0, 40.0, 40.0],
                        [50.0, 50.0, 60.0, 60.0],
                    ]
                ),
                "id": np.array([1, 2, 3]),
            },
            {
                "bbox": np.array(
                    [
                        [10.0, 10.0, 20.0, 20.0],
                        [30.0, 30.0, 40.0, 40.0],
                        [50.0, 50.0, 60.0, 60.0],
                    ]
                ),
                "id": np.array([11, 12, 13]),
            },
            {1: 11, 12: 2, 3: np.nan},
            [1.0, 3, 0, 0, 0],
        ),
        # ID switch
        (
            {
                "bbox": np.array(
                    [
                        [10.0, 10.0, 20.0, 20.0],
                        [30.0, 30.0, 40.0, 40.0],
                        [50.0, 50.0, 60.0, 60.0],
                    ]
                ),
                "id": np.array([1, 2, 3]),
            },
            {
                "bbox": np.array(
                    [
                        [10.0, 10.0, 20.0, 20.0],
                        [30.0, 30.0, 40.0, 40.0],
                        [50.0, 50.0, 60.0, 60.0],
                    ]
                ),
                "id": np.array([11, 12, 14]),
            },
            {1: 11, 2: 12, 3: 13},
            [2 / 3, 3, 0, 0, 1],
        ),
        # missed detection
        (
            {
                "bbox": np.array(
                    [
                        [10.0, 10.0, 20.0, 20.0],
                        [30.0, 30.0, 40.0, 40.0],
                        [50.0, 50.0, 60.0, 60.0],
                    ]
                ),
                "id": np.array([1, 2, 4]),
            },
            {
                "bbox": np.array(
                    [[10.0, 10.0, 20.0, 20.0], [30.0, 30.0, 40.0, 40.0]]
                ),
                "id": np.array([11, 12]),
            },
            {1: 11, 2: 12, 3: 13},
            [2 / 3, 2, 1, 0, 0],
        ),
        # false positive
        (
            {
                "bbox": np.array(
                    [
                        [10.0, 10.0, 20.0, 20.0],
                        [30.0, 30.0, 40.0, 40.0],
                        [50.0, 50.0, 60.0, 60.0],
                    ]
                ),
                "id": np.array([1, 2, 3]),
            },
            {
                "bbox": np.array(
                    [
                        [10.0, 10.0, 20.0, 20.0],
                        [30.0, 30.0, 40.0, 40.0],
                        [50.0, 50.0, 60.0, 60.0],
                        [70.0, 70.0, 80.0, 80.0],
                    ]
                ),
                "id": np.array([11, 12, 13, 14]),
            },
            {1: 11, 2: 12, 3: 13},
            [2 / 3, 3, 0, 1, 0],
        ),
        # low IOU and ID switch
        (
            {
                "bbox": np.array(
                    [
                        [10.0, 10.0, 20.0, 20.0],
                        [30.0, 30.0, 40.0, 40.0],
                        [50.0, 50.0, 60.0, 60.0],
                    ]
                ),
                "id": np.array([1, 2, 3]),
            },
            {
                "bbox": np.array(
                    [
                        [10.0, 10.0, 20.0, 20.0],
                        [30.0, 30.0, 30.0, 30.0],
                        [50.0, 50.0, 60.0, 60.0],
                    ]
                ),
                "id": np.array([11, 12, 14]),
            },
            {1: 11, 2: 12, 3: 13},
            [0, 2, 1, 1, 1],
        ),
        # low IOU and ID switch on same box
        (
            {
                "bbox": np.array(
                    [
                        [10.0, 10.0, 20.0, 20.0],
                        [30.0, 30.0, 40.0, 40.0],
                        [50.0, 50.0, 60.0, 60.0],
                    ]
                ),
                "id": np.array([1, 2, 3]),
            },
            {
                "bbox": np.array(
                    [
                        [10.0, 10.0, 20.0, 20.0],
                        [30.0, 30.0, 30.0, 30.0],
                        [50.0, 50.0, 60.0, 60.0],
                    ]
                ),
                "id": np.array([11, 14, 13]),
            },
            {1: 11, 2: 12, 3: 13},
            [1 / 3, 2, 1, 1, 0],
        ),
        # current tracked id = prev tracked id, but prev_gt_id != current gt id
        (
            {
                "bbox": np.array(
                    [
                        [10.0, 10.0, 20.0, 20.0],
                        [30.0, 30.0, 40.0, 40.0],
                        [50.0, 50.0, 60.0, 60.0],
                    ]
                ),
                "id": np.array([1, 2, 4]),
            },
            {
                "bbox": np.array(
                    [
                        [10.0, 10.0, 20.0, 20.0],
                        [30.0, 30.0, 40.0, 40.0],
                        [50.0, 50.0, 60.0, 60.0],
                    ]
                ),
                "id": np.array([11, 12, 13]),
            },
            {1: 11, 2: 12, 3: 13},
            [2 / 3, 3, 0, 0, 1],
        ),
        # ID swapped
        (
            {
                "bbox": np.array(
                    [
                        [10.0, 10.0, 20.0, 20.0],
                        [30.0, 30.0, 40.0, 40.0],
                        [50.0, 50.0, 60.0, 60.0],
                    ]
                ),
                "id": np.array([1, 2, 3]),
            },
            {
                "bbox": np.array(
                    [
                        [10.0, 10.0, 20.0, 20.0],
                        [30.0, 30.0, 40.0, 40.0],
                        [50.0, 50.0, 60.0, 60.0],
                    ]
                ),
                "id": np.array([11, 13, 12]),
            },
            {1: 11, 2: 12, 3: 13},
            [1 / 3, 3, 0, 0, 2],
        ),
    ],
)
def test_evaluate_mota(
    gt_data,
    pred_data,
    prev_frame_id_map,
    expected_output,
    tracker_evaluate_interface,
):
    (
        mota,
        true_positives,
        missed_detections,
        false_positives,
        num_switches,
        total_gt,
        _,
    ) = tracker_evaluate_interface.evaluate_mota(
        gt_data,
        pred_data,
        0.1,  # iou_threshold
        prev_frame_id_map,
    )
    assert mota == pytest.approx(expected_output[0])
    assert true_positives == expected_output[1]
    assert missed_detections == expected_output[2]
    assert false_positives == expected_output[3]
    assert num_switches == expected_output[4]
    assert total_gt == (true_positives + missed_detections)
