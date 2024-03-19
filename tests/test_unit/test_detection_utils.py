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


@pytest.mark.parametrize(
    "box1, box2, expected_iou",
    [
        ([0, 0, 10, 10], [5, 5, 12, 12], 0.25),  # Partial overlap
        ([0, 0, 10, 10], [0, 0, 10, 10], 1.0),   # Identical boxes
        ([0, 0, 10, 10], [20, 20, 30, 30], 0.0), # No overlap
        ([0, 0, 10, 10], [5, 15, 15, 25], 0.0),  # Boxes are disjoint
    ]
)
def test_calculate_iou(box1, box2, expected_iou):
    # Convert lists to numpy arrays
    box1 = np.array(box1)
    box2 = np.array(box2)

    # Calculate IoU
    iou = calculate_iou(box1, box2)
    print(iou)

    # Check if IoU matches expected value
    assert iou == pytest.approx(expected_iou, abs=1e-2)


# def test_count_identity_switches():
#     # Test with no previous frame
#     prev_frame = None
#     current_frame = [[1, 2], [3, 4], [5, 6]]
#     assert count_identity_switches(prev_frame, current_frame) == 0

#     # Test with no identity switches
#     prev_frame = [[1, 2], [3, 4], [5, 6]]
#     current_frame = [[1, 2], [3, 4], [5, 6]]
#     assert count_identity_switches(prev_frame, current_frame) == 0

#     # Test with one identity switch
#     prev_frame = [[1, 2], [3, 4], [5, 6]]
#     current_frame = [[1, 2], [5, 6], [3, 4]]
#     assert count_identity_switches(prev_frame, current_frame) == 1

#     # Test with multiple identity switches
#     prev_frame = [[1, 2], [3, 4], [5, 6]]
#     current_frame = [[1, 2], [5, 6], [7, 8], [9, 10], [3, 4]]
#     assert count_identity_switches(prev_frame, current_frame) == 2

#     # Test with empty frames
#     prev_frame = []
#     current_frame = []
#     assert count_identity_switches(prev_frame, current_frame) == 0

#     # Test with different object IDs but same bounding boxes
#     prev_frame = [[1, 2], [3, 4], [5, 6]]
#     current_frame = [[7, 8], [9, 10], [11, 12]]
#     assert count_identity_switches(prev_frame, current_frame) == 0

#     # Test with one object ID missing in the current frame
#     prev_frame = [[1, 2], [3, 4], [5, 6]]
#     current_frame = [[1, 2], [3, 4]]
#     assert count_identity_switches(prev_frame, current_frame) == 0

#     # Test with one object ID missing in the previous frame
#     prev_frame = [[1, 2], [3, 4]]
#     current_frame = [[1, 2], [3, 4], [5, 6]]
#     assert count_identity_switches(prev_frame, current_frame) == 0
    

def test_get_ground_truth_data_with_test_csv():
    test_csv_file = "tests/data/gt_test.csv"

    gt_data = get_ground_truth_data(test_csv_file)

    assert len(gt_data) == 2

    for i, frame_data in enumerate(gt_data):
        for j, detection_data in enumerate(frame_data):
            assert detection_data.shape == (5,), f"Detection data shape mismatch for frame {i}"

    expected_ids = [2.0, 1.0]
    for i, frame_data in enumerate(gt_data):
        for j, detection_data in enumerate(frame_data):
            assert detection_data[4] == expected_ids[j], f"Failed for frame {i}, detection {j}"
