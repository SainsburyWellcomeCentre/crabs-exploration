import random
from pathlib import Path

import pytest
import torch
import torchvision.transforms.v2 as transforms
import yaml  # type: ignore

from crabs.detection.datamodules import CrabsDataModule

DEFAULT_CONFIG = (
    Path(__file__).parents[2]
    / "crabs"
    / "detection"
    / "config"
    / "faster_rcnn.yaml"
)


@pytest.fixture
def default_train_config():
    config_file = DEFAULT_CONFIG
    with open(config_file, "r") as f:
        return yaml.safe_load(f)


@pytest.fixture
def expected_data_augm_transforms():
    return transforms.Compose(
        [
            transforms.ToImage(),
            transforms.GaussianBlur(kernel_size=[5, 9], sigma=[0.1, 5.0]),
            transforms.ColorJitter(brightness=(0.5, 1.5), hue=(-0.3, 0.3)),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomRotation(
                degrees=[-10.0, 10.0],
                interpolation=transforms.InterpolationMode.NEAREST,
                expand=False,
                fill=0,
            ),
            transforms.RandomAdjustSharpness(p=0.5, sharpness_factor=0.5),
            transforms.RandomAutocontrast(p=0.5),
            transforms.RandomEqualize(p=0.5),
            transforms.ClampBoundingBoxes(),
            transforms.SanitizeBoundingBoxes(min_size=1.0, labels_getter=None),
            transforms.ToDtype(torch.float32, scale=True),
        ]
    )


@pytest.fixture
def expected_no_data_augm_transforms():
    return transforms.Compose(
        [
            transforms.ToImage(),
            transforms.ToDtype(torch.float32, scale=True),
        ]
    )


@pytest.fixture
def crabs_data_module_with_data_augm(default_train_config):
    return CrabsDataModule(
        list_img_dirs=["dir1", "dir2"],
        list_annotation_files=["anno1", "anno2"],
        config=default_train_config,
        split_seed=123,
        no_data_augmentation=False,
    )


@pytest.fixture
def crabs_data_module_without_data_augm(default_train_config):
    return CrabsDataModule(
        list_img_dirs=["dir1", "dir2"],
        list_annotation_files=["anno1", "anno2"],
        config=default_train_config,
        split_seed=123,
        no_data_augmentation=True,
    )


def compare_transforms_attrs_excluding(transform1, transform2, keys_to_skip):
    """Compare the attributes of two transforms excluding those in list."""

    transform1_attrs_without_fns = {
        key: val
        for key, val in transform1.__dict__.items()
        if key not in keys_to_skip
    }

    transform2_attrs_without_fns = {
        key: val
        for key, val in transform2.__dict__.items()
        if key not in keys_to_skip
    }

    return transform1_attrs_without_fns == transform2_attrs_without_fns


@pytest.mark.parametrize(
    "crabs_data_module, expected_train_transforms",
    [
        ("crabs_data_module_with_data_augm", "expected_data_augm_transforms"),
        (
            "crabs_data_module_without_data_augm",
            "expected_no_data_augm_transforms",
        ),
    ],
)
def test_get_train_transform(
    crabs_data_module, expected_train_transforms, request
):
    crabs_data_module = request.getfixturevalue(crabs_data_module)
    expected_train_transforms = request.getfixturevalue(
        expected_train_transforms
    )

    train_transforms = crabs_data_module._get_train_transform()

    assert isinstance(train_transforms, transforms.Compose)

    # assert all transforms in Compose have same attributes
    for train_tr, expected_train_tr in zip(
        train_transforms.transforms,
        expected_train_transforms.transforms,
    ):
        # we skip the attribute `_labels_getter` of `SanitizeBoundingBoxes`
        # because it points to a lambda function, which does not have a comparison defined.
        assert compare_transforms_attrs_excluding(
            transform1=train_tr,
            transform2=expected_train_tr,
            keys_to_skip=["_labels_getter"],
        )


@pytest.mark.parametrize(
    "crabs_data_module, expected_test_val_transforms",
    [
        (
            "crabs_data_module_with_data_augm",
            "expected_no_data_augm_transforms",
        ),
        (
            "crabs_data_module_without_data_augm",
            "expected_no_data_augm_transforms",
        ),
    ],
)
def test_get_test_val_transform(
    crabs_data_module, expected_test_val_transforms, request
):
    crabs_data_module = request.getfixturevalue(crabs_data_module)
    expected_test_val_transforms = request.getfixturevalue(
        expected_test_val_transforms
    )

    test_val_transforms = crabs_data_module._get_test_val_transform()

    assert isinstance(test_val_transforms, transforms.Compose)

    # assert all transforms in Compose have same attributes
    for test_val_tr, expected_test_val_tr in zip(
        test_val_transforms.transforms,
        expected_test_val_transforms.transforms,
    ):
        assert test_val_tr.__dict__ == expected_test_val_tr.__dict__


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


@pytest.mark.parametrize(
    "crabs_data_module",
    [
        "crabs_data_module_with_data_augm",
        "crabs_data_module_without_data_augm",
    ],
)
def test_collate_fn(crabs_data_module, dummy_dataset, request):
    crabs_data_module = request.getfixturevalue(crabs_data_module)
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
