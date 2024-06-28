import re
from unittest.mock import MagicMock, patch

import numpy as np
import pytest
import torch

from crabs.detector.utils.visualization import (
    draw_bbox,
    draw_detection,
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
    "output_dir_name, expected_dir_name",
    [("output", r"^output$"), ("", r"^results_\d{8}_\d{6}$")],
)
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
@patch("crabs.detector.utils.visualization.cv2.imwrite")
@patch("crabs.detector.utils.visualization.os.makedirs")
def test_save_images_with_boxes(
    mock_makedirs, mock_imwrite, detections, output_dir_name, expected_dir_name
):
    trained_model = MagicMock()
    test_dataloader = MagicMock()
    trained_model.return_value = detections

    save_images_with_boxes(
        test_dataloader,
        trained_model,
        output_dir=output_dir_name,
        score_threshold=0.5,
    )

    # extract and check first positional argument to (mocked) os.makedirs
    input_path_makedirs = mock_makedirs.call_args[0][0]
    assert re.match(expected_dir_name, input_path_makedirs)

    assert mock_imwrite.call_count == len(test_dataloader)
