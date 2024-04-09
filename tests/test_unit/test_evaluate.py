from unittest.mock import MagicMock, patch

import pytest
import torch

from crabs.detection_tracking.evaluate import (
    compute_confusion_matrix_elements,
    compute_precision_recall,
    save_images_with_boxes,
)


@pytest.fixture
def test_dataloader():
    return MagicMock()


@pytest.fixture
def trained_model():
    return MagicMock()


@pytest.fixture
def score_threshold():
    return 0.5


@patch("cv2.imwrite")
@patch("os.makedirs")
@patch("crabs.detection_tracking.detection_utils.draw_detection")
def test_save_images_with_boxes(
    mock_draw_detection,
    mock_makedirs,
    mock_imwrite,
    test_dataloader,
    trained_model,
    score_threshold,
):
    detections = MagicMock()
    mock_draw_detection.return_value = MagicMock()
    trained_model.return_value = detections

    save_images_with_boxes(
        test_dataloader,
        trained_model,
        score_threshold,
    )

    assert mock_makedirs.called_once_with("results", exist_ok=True)
    assert mock_draw_detection.call_count == len(test_dataloader)
    assert mock_imwrite.call_count == len(test_dataloader)


def test_compute_precision_recall():
    class_stats = {"crab": {"tp": 10, "fp": 2, "fn": 1}}

    precision, recall, _ = compute_precision_recall(class_stats)

    assert precision == 10 / max(10 + 2, 1)
    assert recall == 10 / max(10 + 1, 1)


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
