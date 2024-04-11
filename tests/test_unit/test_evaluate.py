import torch

from crabs.detection_tracking.evaluate import (
    compute_confusion_matrix_elements,
    compute_precision_recall,
)


def test_compute_precision_recall():
    class_stats = {"crab": {"tp": 10, "fp": 2, "fn": 1}}

    precision, recall, _ = compute_precision_recall(class_stats)

    assert precision == 10 / max(class_stats['crab']['tp'] + class_stats['crab']['fp'], class_stats['crab']['fn'])
    assert recall == 10 / max(class_stats['crab']['tp'] + class_stats['crab']['fn'], class_stats['crab']['fn'])


def test_compute_precision_recall_zero_division():
    class_stats = {"crab": {"tp": 0, "fp": 0, "fn": 0}}

    precision, recall, _ = compute_precision_recall(class_stats)

    # Assert expected precision and recall values when all counts are zero
    assert precision == 0
    assert recall == 0


def test_compute_confusion_matrix_elements():
    targets = [
        {
            "image_id": 1,
            "boxes": torch.tensor(
                [[27, 11, 63, 33], [87, 4, 118, 23], [154, 152, 192, 164]]
            ),
            "labels": torch.tensor([1, 1, 1]),
        }
    ]
    detections = [
        {
            "boxes": torch.tensor([[27, 11, 63, 33], [90, 4, 200, 23]]),
            "labels": torch.tensor(
                [
                    1,
                    1,
                ]
            ),
            "scores": torch.tensor([0.5083, 0.4805]),
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
