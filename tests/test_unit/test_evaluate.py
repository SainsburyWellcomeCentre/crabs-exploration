from unittest.mock import patch

import pytest
import torch

from crabs.detector.utils.evaluate import (
    compute_precision_recall,
    evaluate_detections_hungarian,
)


def test_evaluate_detections_hungarian():
    # sample boxes  xyxy
    box_1 = torch.tensor([27, 11, 63, 33])
    box_2 = torch.tensor([87, 4, 118, 23])
    box_3 = torch.tensor([154, 152, 192, 164])
    delta_box = torch.tensor([3, 0, 82, 0])

    ious_threshold = 0.5

    gt_dicts_batch = [
        {
            "image_id": 1,
            "boxes": torch.vstack((box_1, box_2, box_3)),
            "labels": torch.tensor([1, 1, 1]),
        }
    ]
    detections_dicts_batch = [
        {
            "boxes": torch.vstack((box_1, box_2 + delta_box)),
            "labels": torch.tensor([1, 1]),
            "scores": torch.tensor([0.5083, 0.4805]),
        }
    ]

    results = evaluate_detections_hungarian(
        detections_dicts_batch, gt_dicts_batch, ious_threshold
    )

    assert results["tp"] == 1
    assert results["fp"] == 1
    assert results["fn"] == 2


@pytest.mark.parametrize(
    "mocked_results,expected_precision_recall",
    [
        (
            {"tp": 10, "fp": 2, "fn": 1},
            {"precision": 10 / 12, "recall": 10 / 11},
        ),
        (
            {"tp": 0, "fp": 0, "fn": 0},
            {"precision": 0.0, "recall": 0.0},
            # all counts are zero
        ),
    ],
)
def test_compute_precision_recall(mocked_results, expected_precision_recall):
    # Mock the evaluate_detections_hungarian function
    with patch(
        "crabs.detector.utils.evaluate.evaluate_detections_hungarian"
    ) as mock_evaluate:
        # mock the return value of `evaluate_detections_hungarian`
        # (called by `compute_precision_recall`)
        mock_evaluate.return_value = mocked_results

        # compute precision and recall
        precision, recall = compute_precision_recall([], [], 0)

    assert precision == expected_precision_recall["precision"]
    assert recall == expected_precision_recall["recall"]
