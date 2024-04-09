import os
from pathlib import Path

import cv2
import numpy as np
import pytest
import torch

from crabs.detection_tracking.detection_utils import (
    coco_category,
    draw_bbox,
    draw_detection,
    prep_annotation_files,
    prep_img_directories,
    save_model,
)


def test_coco_category():
    expected_categories = ["__background__", "crab"]
    assert coco_category() == expected_categories


def test_save_model(tmpdir):
    model = torch.nn.Linear(10, 2)
    save_model(model)
    assert os.path.exists(tmpdir)


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
def dataset_dirs(tmp_path):
    dataset1_dir = tmp_path / "dataset1"
    dataset1_dir.mkdir()
    (dataset1_dir / "frames").mkdir()

    dataset2_dir = tmp_path / "dataset2"
    dataset2_dir.mkdir()
    (dataset2_dir / "frames").mkdir()

    return [str(dataset1_dir), str(dataset2_dir)]


def test_prep_img_directories(dataset_dirs):
    expected_result = [
        str(Path(dataset_dirs[0]) / "frames"),
        str(Path(dataset_dirs[1]) / "frames"),
    ]
    assert prep_img_directories(dataset_dirs) == expected_result


def test_prep_annotation_files_default(dataset_dirs):
    result = prep_annotation_files([], dataset_dirs)
    print(result)
    expected_result = [
        str(
            Path(dataset_dirs[0])
            / "annotations"
            / "VIA_JSON_combined_coco_gen.json"
        ),
        str(
            Path(dataset_dirs[1])
            / "annotations"
            / "VIA_JSON_combined_coco_gen.json"
        ),
    ]
    assert result == expected_result


def test_prep_annotation_files_with_filenames(dataset_dirs):
    input_annotation_files = ["file1.json", "file2.json"]
    result = prep_annotation_files(input_annotation_files, dataset_dirs)
    expected_result = [
        str(Path(dataset_dirs[0]) / "annotations" / "file1.json"),
        str(Path(dataset_dirs[1]) / "annotations" / "file2.json"),
    ]
    assert result == expected_result


def test_prep_annotation_files_with_full_paths(dataset_dirs):
    input_annotation_files = ["/path/to/file1.json", "/path/to/file2.json"]
    result = prep_annotation_files(input_annotation_files, dataset_dirs)
    assert result == input_annotation_files
