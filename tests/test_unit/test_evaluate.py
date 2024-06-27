import torch

<<<<<<< HEAD
from crabs.detection.evaluate import (
=======
from crabs.detection_tracking.evaluate_utils import (
>>>>>>> e18357e44d9c069ea617d3eaf41da3d2389812dd
    compute_confusion_matrix_elements,
    compute_precision_recall,
)


def test_compute_precision_recall():
    class_stats = {"crab": {"tp": 10, "fp": 2, "fn": 1}}

    precision, recall, _ = compute_precision_recall(class_stats)

    assert precision == class_stats["crab"]["tp"] / max(
        class_stats["crab"]["tp"] + class_stats["crab"]["fp"],
        class_stats["crab"]["fn"],
    )
    assert recall == class_stats["crab"]["tp"] / max(
        class_stats["crab"]["tp"] + class_stats["crab"]["fn"],
        class_stats["crab"]["fn"],
    )


def test_compute_precision_recall_zero_division():
    class_stats = {"crab": {"tp": 0, "fp": 0, "fn": 0}}

    precision, recall, _ = compute_precision_recall(class_stats)

    # Assert expected precision and recall values when all counts are zero
    assert precision == max(
        class_stats["crab"]["tp"] + class_stats["crab"]["fp"],
        class_stats["crab"]["fn"],
    )
    assert recall == max(
        class_stats["crab"]["tp"] + class_stats["crab"]["fp"],
        class_stats["crab"]["fn"],
    )


def test_compute_confusion_matrix_elements():
    # ground truth
    gt_box_1 = torch.tensor([27, 11, 63, 33])
    gt_box_2 = torch.tensor([87, 4, 118, 23])
    gt_box_3 = torch.tensor([154, 152, 192, 164])
    gt_labels = torch.tensor([1, 1, 1])
    # predictions
    delta_box_2 = torch.tensor([3, 0, 82, 0])
    detection_labels = torch.tensor([1, 1])
    detection_scores = torch.tensor([0.5083, 0.4805])
    targets = [
        {
            "image_id": 1,
            "boxes": torch.vstack((gt_box_1, gt_box_2, gt_box_3)),
            "labels": gt_labels,
        }
    ]
    detections = [
        {
            "boxes": torch.vstack((gt_box_1, gt_box_2 + delta_box_2)),
            "labels": detection_labels,
            "scores": detection_scores,
        }
    ]
    ious_threshold = 0.5

    precision, recall, class_stats = compute_confusion_matrix_elements(
        targets, detections, ious_threshold
    )

    assert precision == 0.5
    assert recall == 1 / 3
    assert class_stats["crab"]["tp"] == 1
    assert class_stats["crab"]["fp"] == 1
    assert class_stats["crab"]["fn"] == 2
