import pytest
import torch
import torchvision.transforms.v2 as transforms

from crabs.detection_tracking.datamodules import CrabsDataModule


@pytest.fixture
def crabs_data_module():
    # Define a test config dictionary
    test_config = {
        "train_fraction": 0.8,
        "val_over_test_fraction": 0,
        "transform_brightness": 0.5,
        "transform_hue": 0.3,
        "gaussian_blur_params": {"kernel_size": [5, 9], "sigma": [0.1, 5.0]},
    }
    return CrabsDataModule(
        list_img_dirs=["dir1", "dir2"],
        list_annotation_files=["anno1", "anno2"],
        config=test_config,
        split_seed=123,
    )


def test_get_train_transform(crabs_data_module):
    train_transform = crabs_data_module._get_train_transform()
    assert isinstance(train_transform, transforms.Compose)

    expected_transforms = [
        transforms.ToImage(),
        transforms.ColorJitter(brightness=0.5, hue=0.3),
        transforms.GaussianBlur(kernel_size=[5, 9], sigma=[0.1, 5.0]),
        transforms.ToDtype(torch.float32, scale=True),
    ]

    assert len(train_transform.transforms) == len(expected_transforms)
    for i, expected_transform in enumerate(expected_transforms):
        assert isinstance(
            train_transform.transforms[i], type(expected_transform)
        )


def test_get_test_transform(crabs_data_module):
    test_transform = crabs_data_module._get_test_val_transform()

    assert isinstance(test_transform, transforms.Compose)

    expected_transforms = [
        transforms.ToImage(),
        transforms.ToDtype(torch.float32, scale=True),
    ]

    assert len(test_transform.transforms) == len(expected_transforms)
    for i, expected_transform in enumerate(expected_transforms):
        assert isinstance(
            test_transform.transforms[i], type(expected_transform)
        )
