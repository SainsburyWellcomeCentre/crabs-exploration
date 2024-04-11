import random

import pytest
import torch
import torchvision.transforms.v2 as transforms

from crabs.detection_tracking.datamodules import CrabsDataModule


@pytest.fixture
def train_config():
    return {
        "train_fraction": 0.8,
        "val_over_test_fraction": 0,
        "transform_brightness": 0.5,
        "transform_hue": 0.3,
        "gaussian_blur_params": {"kernel_size": [5, 9], "sigma": [0.1, 5.0]},
    }


@pytest.fixture
def crabs_data_module(train_config):
    return CrabsDataModule(
        list_img_dirs=["dir1", "dir2"],
        list_annotation_files=["anno1", "anno2"],
        config=train_config,
        split_seed=123,
    )


@pytest.fixture
def transforms_train_set(train_config):
    return [
        transforms.ToImage(),
        transforms.ColorJitter(
            brightness=train_config["transform_brightness"],
            hue=train_config["transform_hue"],
        ),
        transforms.GaussianBlur(
            kernel_size=train_config["gaussian_blur_params"]["kernel_size"],
            sigma=train_config["gaussian_blur_params"]["sigma"],
        ),
        transforms.ToDtype(torch.float32, scale=True),
    ]


def test_get_train_transform(crabs_data_module, expected_transforms_train_set):
    train_transform = crabs_data_module._get_train_transform()
    assert isinstance(train_transform, transforms.Compose)

    assert len(train_transform.transforms) == len(expected_transforms_train_set)
    for transform, expected_transform in zip(train_transform.transforms, expected_transforms_train_set):
        assert isinstance(transform, type(expected_transform))


@pytest.fixture
def transforms_test_set():
    return [
        transforms.ToImage(),
        transforms.ToDtype(torch.float32, scale=True),
    ]


def test_get_test_transform(crabs_data_module, transforms_test_set):
    test_transform = crabs_data_module._get_test_val_transform()
    assert isinstance(test_transform, transforms.Compose)

    assert len(test_transform.transforms) == len(transforms_test_set)
    for i, expected_transform in enumerate(transforms_test_set):
        assert isinstance(
            test_transform.transforms[i], type(expected_transform)
        )


@pytest.fixture
def dummy_dataset():
    """Create dummy images and annotations for testing."""
    num_samples = 5
    images = [torch.randn(3, 256, 256) for _ in range(num_samples)]
    annotations = []
    for _ in range(num_samples):
        # Generate random number of bounding boxes for each image
        num_boxes = random.randint(1, 5)
        boxes = []
        for _ in range(num_boxes):
            # Generate random bounding box coordinates within image size
            x_min = random.randint(0, 200)
            y_min = random.randint(0, 200)
            x_max = random.randint(x_min + 10, 256)
            y_max = random.randint(y_min + 10, 256)
            boxes.append([x_min, y_min, x_max, y_max])
        annotations.append(torch.tensor(boxes))
    return images, annotations


def test_collate_fn(crabs_data_module, dummy_dataset):
    collated_data = crabs_data_module._collate_fn(dummy_dataset)

    assert len(collated_data) == len(dummy_dataset[0])  # images
    assert len(collated_data) == len(dummy_dataset[1])  # annotations

    for i, sample in enumerate(collated_data):
        # check length
        assert len(sample) == 2

        # check same content as in dummy dataset
        image, annotation = sample
        assert torch.equal(image, dummy_dataset[0][i])
        assert torch.equal(annotation, dummy_dataset[1][i])
