from unittest.mock import MagicMock, patch

import pytest
import torch

from crabs.detection_tracking.evaluate import (
    compute_precision_recall,
    save_images_with_boxes,
    compute_precision_recall
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


@pytest.fixture
def device():
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


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
    device,
):
    detections = MagicMock()
    mock_draw_detection.return_value = MagicMock()
    trained_model.return_value = detections

    save_images_with_boxes(
        test_dataloader, trained_model, score_threshold, device
    )

    assert mock_makedirs.called_once_with("results", exist_ok=True)
    assert mock_draw_detection.call_count == len(test_dataloader)
    assert mock_imwrite.call_count == len(test_dataloader)


@pytest.fixture
def class_stats():
    return {
        "crab": {"tp": 10, "fp": 5, "fn": 3},
    }


def test_compute_precision_recall():
    class_stats = {
        "class1": {"tp": 10, "fp": 2, "fn": 1},
        "class2": {"tp": 15, "fp": 3, "fn": 2}
    }

    precision, recall = compute_precision_recall(class_stats)

    # Assert expected precision and recall values for each class
    assert precision["class1"] == 10 / max(10 + 2, 1)
    assert precision["class2"] == 15 / max(15 + 3, 1)
    assert recall["class1"] == 10 / max(10 + 1, 1)
    assert recall["class2"] == 15 / max(15 + 2, 1)


def test_compute_precision_recall_zero_division():
    class_stats = {
        "class1": {"tp": 0, "fp": 0, "fn": 0}
    }

    precision, recall = compute_precision_recall(class_stats)

    # Assert expected precision and recall values when all counts are zero
    assert precision["class1"] == 0
    assert recall["class1"] == 0
