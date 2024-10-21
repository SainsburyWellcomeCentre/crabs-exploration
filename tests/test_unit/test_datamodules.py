import random
from pathlib import Path

import pytest
import torch
import torchvision.transforms.v2 as transforms
import yaml  # type: ignore
from torchvision.utils import save_image

from crabs.detector.datamodules import CrabsDataModule

DEFAULT_CONFIG = (
    Path(__file__).parents[2]
    / "crabs"
    / "detector"
    / "config"
    / "faster_rcnn.yaml"
)


@pytest.fixture
def default_train_config():
    config_file = DEFAULT_CONFIG
    with open(config_file, "r") as f:
        return yaml.safe_load(f)


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


@pytest.fixture
def dummy_dataset():
    """Create dummy images and annotations for testing.

    The dataset consists of 5 images, with a random number of bounding boxes
    per image. The bounding boxes have fixed width and height, but their location
    is randomised.
    """
    n_images = 5
    img_size = 256
    fixed_width_height = 10

    images = [torch.randn(3, img_size, img_size) for _ in range(n_images)]
    annotations = []
    for _ in range(n_images):
        # Generate random number of bounding boxes for each image
        n_bboxes = random.randint(1, 5)
        boxes = []
        for _ in range(n_bboxes):
            # Randomise the location of the top left corner of the bounding box
            x_min = random.randint(0, img_size - fixed_width_height)
            y_min = random.randint(0, img_size - fixed_width_height)

            # Add fixed width and height to get the bottom right corner
            x_max = x_min + fixed_width_height
            y_max = y_min + fixed_width_height
            boxes.append([x_min, y_min, x_max, y_max])
        annotations.append(torch.tensor(boxes))
    return images, annotations


@pytest.fixture(scope="session")
def dummy_dataset_dirs(dummy_dataset, tmp_path_factory):
    """Save dummy dataset to temporary directories and return their paths."""

    # Get dummy data
    images, annotations = dummy_dataset

    # Create temporary directories
    frames_path = tmp_path_factory.mktemp("frames")
    annotations_path = tmp_path_factory.mktemp("annotations")

    # Save images to temporary directory
    for idx, img in enumerate(images):
        out_path = frames_path / f"frame_{idx:04d}.png"
        save_image(img, out_path)

    # Save annotations with expected format to temporary directory

    # return as dict
    dataset_dict = {
        "frames": frames_path,
        "annotations": annotations_path,
    }

    return dataset_dict


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
    """Test transforms linked to training set are as expected"""
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
    """Test transforms linked to test and validation sets are as expected"""
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


@pytest.mark.parametrize(
    "seed, expected_indices",
    [
        (123, {"train": [1, 2, 3], "test": [1, 2, 3], "val": [1, 2, 3]}),
        (42, {"train": [1, 2, 3], "test": [1, 2, 3], "val": [1, 2, 3]}),
    ],
)
def test_compute_splits(
    seed,
    expected_indices,
    dummy_dataset_dirs,
    default_train_config,  # ---- edit config too?
):
    """Test dataset splits are reproducible and according to the requested fraction"""

    # Create datamodule
    dm = CrabsDataModule(
        list_img_dirs=[dummy_dataset_dirs["frames"]],
        list_annotation_files=[dummy_dataset_dirs["annotations"]],
        config=default_train_config,
        split_seed=seed,
        no_data_augmentation=False,
    )

    # Add transforms
    train_transform = dm._get_test_val_transform()
    test_and_val_transform = dm._get_test_val_transform()

    # Compute splits
    train_dataset, _, _ = dm._compute_splits(train_transform)
    _, test_dataset, val_dataset = dm._compute_splits(test_and_val_transform)

    # Check split sizes are as expected
    total_dataset_size = (
        len(train_dataset) + len(test_dataset) + len(val_dataset)
    )
    assert total_dataset_size == 50
    # TODO: change to np.isclose
    assert len(train_dataset) / total_dataset_size == 0.8
    assert len(test_dataset) / total_dataset_size == 0.1
    assert len(val_dataset) / total_dataset_size == 0.1

    # Check splits are non-overlapping in image IDs
    # --- I cannot do this because samples are tuple(image, annotation)
    # assert len(set(train_dataset) & set(test_dataset)) == 0
    # assert len(set(train_dataset) & set(val_dataset)) == 0
    # assert len(set(test_dataset) & set(val_dataset)) == 0
    # assert len(set(train_dataset) & set(test_dataset)) == 0

    # Compute lists of image IDs per dataset
    image_ids_per_dataset = {}
    for dataset, dataset_str in zip(
        [train_dataset, test_dataset, val_dataset], ["train", "test", "val"]
    ):
        image_ids_per_dataset[dataset_str] = [
            sample[1]["image_id"] for sample in dataset
        ]

    # Check splits are non-overlapping in image IDs
    # TODO: Can I improve this? it is v slow!
    assert (
        len(
            set(image_ids_per_dataset["train"])
            & set(image_ids_per_dataset["test"])
        )
        == 0
    )
    assert (
        len(
            set(image_ids_per_dataset["train"])
            & set(image_ids_per_dataset["val"])
        )
        == 0
    )
    assert (
        len(
            set(image_ids_per_dataset["test"])
            & set(image_ids_per_dataset["val"])
        )
        == 0
    )

    # Check splits are reproducible
    # we check we always get the same indices from the dataset
    # we input to `random_split` given the same seed
    # Note that the indices are not the same as the image IDs!
    assert train_dataset.indices[:3] == expected_indices["train"]
    assert test_dataset.indices[:3] == expected_indices["test"]
    assert val_dataset.indices[:3] == expected_indices["val"]
