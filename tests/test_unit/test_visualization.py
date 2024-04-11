from unittest.mock import MagicMock, patch

import numpy as np
import pytest
import torch

from crabs.detection_tracking.visualization import (
    draw_bbox,
    draw_detection,
    save_images_with_boxes,
)


@pytest.fixture
def sample_image():
    # Create a sample image for testing
    return np.zeros((100, 100, 3), dtype=np.uint8)


def test_draw_bbox(sample_image):
    top_left = (10, 10)
    bottom_right = (50, 50)
    color = (0, 255, 0)

    draw_bbox(sample_image, top_left, bottom_right, color)

    # Check if bounding box is drawn
    assert np.any(
        sample_image[
            top_left[1] : bottom_right[1], top_left[0] : bottom_right[0]
        ]
        == color
    )


def test_draw_bbox_with_label(sample_image):
    top_left = (10, 10)
    bottom_right = (50, 50)
    color = (0, 255, 0)
    label_text = "Test Label"

    draw_bbox(sample_image, top_left, bottom_right, color, label_text)

    # Check if label text is drawn
    assert np.any(sample_image[10:20, 10:100] == color)


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


@patch("cv2.imwrite")
@patch("os.makedirs")
def test_save_images_with_boxes(
    mock_makedirs,
    mock_imwrite,
):
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
    trained_model = MagicMock()
    test_dataloader = MagicMock()
    trained_model.return_value = detections

    save_images_with_boxes(
        test_dataloader,
        trained_model,
        score_threshold=0.5,
    )

    assert mock_makedirs.called_once_with("results", exist_ok=True)
    assert mock_imwrite.call_count == len(test_dataloader)
