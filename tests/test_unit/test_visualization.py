import csv
import tempfile
from unittest.mock import MagicMock, patch

import numpy as np
import pytest
import torch

from crabs.detection_tracking.visualization import (
    draw_bbox,
    draw_detection,
    read_metrics_from_csv,
    save_images_with_boxes,
)


@pytest.fixture
def sample_image():
    # Create a sample image for testing
    return np.zeros((100, 100, 3), dtype=np.uint8)


@pytest.mark.parametrize(
    "top_left, bottom_right, color",
    [
        ((10, 10), (50, 50), (0, 255, 0)),
    ],
)
def test_draw_bbox(sample_image, top_left, bottom_right, color):
    draw_bbox(sample_image, top_left, bottom_right, color)

    assert np.any(
        sample_image[
            top_left[1] : bottom_right[1], top_left[0] : bottom_right[0]
        ]
        == color
    )  # Check if bounding box is drawn


@pytest.mark.parametrize(
    "top_left, bottom_right, color, label_text",
    [
        ((10, 10), (50, 50), (0, 255, 0), "Test Label"),
    ],
)
def test_draw_bbox_with_label(
    sample_image, top_left, bottom_right, color, label_text
):
    draw_bbox(sample_image, top_left, bottom_right, color, label_text)

    assert np.any(
        sample_image[
            top_left[1] : bottom_right[1], top_left[0] : bottom_right[0]
        ]
        == color
    )


@pytest.mark.parametrize(
    "top_left, bottom_right, color",
    [
        ((10, 10), (15, 15), (0, 255, 0)),
    ],
)
def test_draw_bbox_green(sample_image, top_left, bottom_right, color):
    draw_bbox(sample_image, top_left, bottom_right, color)

    # crop bbox from image
    # Add 1 to include the last row and column of the bounding box
    # slicing in Python excludes the upper bound
    actual_bbox_crop_rgb = sample_image[
        int(top_left[1]) : int(bottom_right[1]) + 1,
        int(top_left[0]) : int(bottom_right[0]) + 1,
    ]

    # build expected result
    bbox_thickness = 2
    crop_size = (
        bottom_right[1] - top_left[1] + 1,
        bottom_right[0] - top_left[0] + 1,
    )

    expected_bbox_crop_g = np.pad(
        np.zeros(
            (
                (crop_size[0] - (2 * bbox_thickness)),
                (crop_size[1] - (2 * bbox_thickness)),
            )
        ),
        ((bbox_thickness,), (bbox_thickness,)),
        constant_values=((255,), (255,)),
    ).astype(np.uint8)

    expected_bbox_crop_g = np.round(expected_bbox_crop_g).astype(np.uint8)

    # append other channels to build rgb
    expected_bbox_crop_rgb = np.stack(
        [
            np.zeros_like(expected_bbox_crop_g),
            expected_bbox_crop_g,
            np.zeros_like(expected_bbox_crop_g),
        ],
        axis=-1,
    )

    assert np.all(expected_bbox_crop_rgb == actual_bbox_crop_rgb)


@pytest.mark.parametrize(
    "annotations, detections",
    [
        (
            [{"boxes": torch.tensor([[10, 10, 20, 20]])}],
            [
                {
                    "scores": torch.tensor([0.9]),
                    "labels": torch.tensor([1]),
                    "boxes": torch.tensor([[30, 30, 40, 40]]),
                }
            ],
        ),
        (
            [
                {"boxes": torch.tensor([[10, 10, 20, 20]])},
                {"boxes": torch.tensor([[50, 50, 60, 60]])},
            ],
            [
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
            ],
        ),
        (
            [
                {"boxes": torch.tensor([[10, 10, 20, 20]])},
                {"boxes": torch.tensor([[50, 50, 60, 60]])},
            ],
            None,
        ),
    ],
)
def test_draw_detection(annotations, detections):
    imgs = [torch.rand(3, 100, 100)]
    image_with_boxes = draw_detection(imgs, annotations, detections)
    assert image_with_boxes is not None


@pytest.mark.parametrize(
    "detections",
    [
        (
            [
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
            ],
        ),
    ],
)
@patch("cv2.imwrite")
@patch("os.makedirs")
def test_save_images_with_boxes(mock_makedirs, mock_imwrite, detections):
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


@pytest.fixture
def sample_csv_data():
    # Create a sample CSV file with some data
    sample_data = [
        {
            "True Positives": "10",
            "Missed Detections": "2",
            "False Positives": "3",
            "Number of Switches": "1",
            "Total Ground Truth": "15",
            "Mota": "0.8",
        },
        {
            "True Positives": "15",
            "Missed Detections": "3",
            "False Positives": "2",
            "Number of Switches": "2",
            "Total Ground Truth": "20",
            "Mota": "0.9",
        },
    ]
    with tempfile.NamedTemporaryFile(mode="w", delete=False) as temp_file:
        writer = csv.DictWriter(temp_file, fieldnames=sample_data[0].keys())
        writer.writeheader()
        writer.writerows(sample_data)
        temp_file_path = temp_file.name
    yield temp_file_path
    # Clean up after the test
    import os

    os.remove(temp_file_path)


def test_read_metrics_from_csv(sample_csv_data):
    (
        true_positives_list,
        missed_detections_list,
        false_positives_list,
        num_switches_list,
        total_ground_truth_list,
        mota_value_list,
    ) = read_metrics_from_csv(sample_csv_data)

    assert true_positives_list == [10, 15]
    assert missed_detections_list == [2, 3]
    assert false_positives_list == [3, 2]
    assert num_switches_list == [1, 2]
    assert total_ground_truth_list == [15, 20]
    assert mota_value_list == [0.8, 0.9]
