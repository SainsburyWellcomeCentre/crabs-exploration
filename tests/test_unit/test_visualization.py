import os
from unittest.mock import MagicMock, patch

import cv2
import numpy as np
import pytest
import torch

from crabs.detection_tracking.visualization import (
    draw_bbox,
    draw_detection,
    save_images_with_boxes,
)


def test_draw_bbox():
    image = np.zeros((100, 100, 3), dtype=np.uint8)
    draw_bbox(image, (10, 10), (20, 20), (0, 255, 0))
    cv2.imwrite("bbox_test.png", image)
    assert os.path.exists("bbox_test.png")


def test_draw_detection():
    # Test with single image, annotation, and detection
    imgs = [torch.rand(3, 100, 100)]
    annotations = [{"boxes": torch.tensor([[10, 10, 20, 20]])}]
    detections = [
        {
            "scores": torch.tensor([0.9]),
            "labels": torch.tensor([1]),
            "boxes": torch.tensor([[30, 30, 40, 40]]),
        }
    ]
    image_with_boxes = draw_detection(imgs, annotations, detections)
    assert image_with_boxes is not None

    # Test with multiple annotations, and detections
    imgs = [torch.rand(3, 100, 100)]
    annotations = [
        {"boxes": torch.tensor([[10, 10, 20, 20]])},
        {"boxes": torch.tensor([[50, 50, 60, 60]])},
    ]
    detections = [
        {
            "scores": torch.tensor([0.9]),
            "labels": torch.tensor([1]),
            "boxes": torch.tensor([[30, 30, 40, 40]]),
        },
        {
            "scores": torch.tensor([0.8]),
            "labels": torch.tensor([2]),
            "boxes": torch.tensor([[70, 70, 80, 80]]),
        },
    ]
    image_with_boxes = draw_detection(imgs, annotations, detections)
    assert image_with_boxes is not None

    # Test with different score thresholds
    image_with_boxes = draw_detection(
        imgs, annotations, detections, score_threshold=0.5
    )
    assert image_with_boxes is not None

    # Test with missing detections
    image_with_boxes = draw_detection(imgs, annotations, None)
    assert image_with_boxes is not None


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
@patch("crabs.detection_tracking.visualization.draw_detection")
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
