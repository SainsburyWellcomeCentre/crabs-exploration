from unittest.mock import MagicMock, patch

import pytest
import torch

from crabs.detection_tracking.evaluate import (
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


@patch("logging.info")
def test_compute_precision_recall(mock_logging_info, class_stats):
    precision, recall = compute_precision_recall(class_stats)

    # Calculate expected precision and recall for "crab" class
    crab_precision = 10 / max(10 + 5, 1)
    crab_recall = 10 / max(10 + 3, 1)

    # # Assertions
    # mock_logging_info.assert_called_once_with(
    #     f"Precision: {crab_precision:.4f}, Recall: {crab_recall:.4f}, "
    #     f"False Positive: 5, False Negative: 3"
    # )
    assert precision == pytest.approx(crab_precision, abs=1e-4)
    assert recall == pytest.approx(crab_recall, abs=1e-4)
