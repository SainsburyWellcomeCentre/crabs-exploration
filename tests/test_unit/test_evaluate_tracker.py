from pathlib import Path

import numpy as np
import pytest

from crabs.tracker.evaluate_tracker import TrackerEvaluate


@pytest.fixture
def evaluation():
    test_csv_file = Path(__file__).parents[1] / "data" / "gt_test.csv"
    return TrackerEvaluate(
        test_csv_file, predicted_boxes_id=[], iou_threshold=0.1
    )


def test_get_ground_truth_data(evaluation):
    ground_truth_dict = evaluation.get_ground_truth_data()

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


def test_ground_truth_data_from_csv(evaluation):
    expected_data = {
        0: {
            "bbox": np.array(
                [
                    [2894.8606, 975.8517, 2945.8606, 1016.8517],
                    [940.6089, 1192.637, 989.6089, 1230.637],
                ],
                dtype=np.float32,
            ),
            "id": np.array([2.0, 1.0], dtype=np.float32),
        },
        1: {
            "bbox": np.array(
                [[940.6089, 1192.637, 989.6089, 1230.637]], dtype=np.float32
            ),
            "id": np.array([2.0], dtype=np.float32),
        },
    }

    ground_truth_dict = evaluation.get_ground_truth_data()

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


@pytest.mark.parametrize(
    "prev_frame_id_map, current_frame_id_map, expected_output",
    [
        (None, {1: 11, 2: 12, 3: 13, 4: 14}, 0),
        ({1: 11, 2: 12, 3: 13, 4: 14}, {1: 11, 2: 12, 3: 13, 4: 14}, 0),
        ({1: 11, 2: 12, 3: 13, 4: 14}, {1: 12, 2: 11, 3: 13, 4: 14}, 2),
        ({1: 11, 2: 12, 3: 13, 4: 14}, {1: 11, 2: 12, 3: 13}, 0),
        ({1: 11, 2: 12, 3: 13, 4: 14}, {1: 11, 2: 12, 3: 13, 5: 14}, 1),
        ({1: 11, 2: 12, 3: 13}, {1: 11, 2: 12, 3: 13, 4: 14}, 0),
        ({1: 11, 2: 12, 3: 13, 4: 14}, {1: 11, 2: 12, 4: 13}, 1),
        ({1: 11, 2: 12, 3: 13, 4: 14}, {1: 11, 2: 12, 3: 13, 4: 15}, 1),
        ({1: 11, 2: 12, 3: 13, 4: 14}, {1: 11, 2: 12, 3: 13, 4: 15, 5: 16}, 1),
        ({3: 23, 4: 100, 1: 11, 2: 21}, {4: 23, 3: 100, 1: 11, 2: 21}, 2),
        ({1: 11, 2: 12, 3: 13}, {2: 12, 3: 14, 4: 13}, 2),
        ({1: 11, 2: 12, 3: 13}, {1: 11, 2: 14, 3: 13}, 1),
        ({1: 11, 2: 12, 3: np.nan}, {1: 11, 2: 12, 4: 13}, 0),
        ({1: 11, 2: 12, 3: 13}, {1: 11, 2: np.nan, 3: 13}, 0),
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
    "gt_data, pred_data, prev_frame_id_map, expected_mota",
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
            1.0,
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
            1.0,
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
            2 / 3,
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
            2 / 3,
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
            2 / 3,
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
            0,
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
            1 / 3,
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
            2 / 3,
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
            1 / 3,
        ),
    ],
)
def test_evaluate_mota(
    gt_data,
    pred_data,
    prev_frame_id_map,
    expected_mota,
    evaluation,
):
    mota, _ = evaluation.evaluate_mota(
        gt_data,
        pred_data,
        0.1,
        prev_frame_id_map,
    )
    assert mota == pytest.approx(expected_mota)
