import os
import cv2
import numpy as np
import torch
import pytest
from crabs.detection_tracking.detection_utils import *


def test_coco_category():
    expected_categories = ["__background__", "crab"]
    assert coco_category() == expected_categories


def test_save_model(tmpdir):
    model = torch.nn.Linear(10, 2)
    save_model(model)
    assert os.path.exists(tmpdir)


def test_draw_bbox():
    image = np.zeros((100, 100, 3), dtype=np.uint8)
    draw_bbox(image, 10, 10, 20, 20, (0, 255, 0))
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
        {"boxes": torch.tensor([[50, 50, 60, 60]])}
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
        }
    ]
    image_with_boxes = draw_detection(imgs, annotations, detections)
    assert image_with_boxes is not None

    # Test with different score thresholds
    image_with_boxes = draw_detection(imgs, annotations, detections, score_threshold=0.5)
    assert image_with_boxes is not None

    # Test with missing detections
    image_with_boxes = draw_detection(imgs, annotations, None)
    assert image_with_boxes is not None

