import json
import random
from pathlib import Path

import numpy as np
import pytest
import torch
import torchvision.transforms.v2 as transforms
import yaml  # type: ignore
from torchvision.utils import save_image

from crabs.detector.datamodules import CrabsDataModule
from crabs.detector.utils.detection import bbox_tensors_to_COCO_dict

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
    with open(config_file) as f:
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


@pytest.fixture()
def create_dummy_dataset():
    """Return a factory of dummy images and annotations for testing.

    The created datasets consist of N images, with a random number of bounding
    boxes per image. The bounding boxes have fixed width and height, but their
    location is randomized. Both images and annotations are torch tensors.
    """

    def _create_dummy_dataset(n_images):
        """Create a dataset with N images and random bounding boxes per image.

        The number of images in the dataset needs to be > 5 to avoid floating
        point errors in the dataset split.
        """
        img_size = 256
        fixed_width_height = 10

        images = [torch.randn(3, img_size, img_size) for _ in range(n_images)]
        annotations = []
        for _ in range(n_images):
            # Generate random number of bounding boxes for each image
            n_bboxes = random.randint(1, 5)
            boxes = []
            for _ in range(n_bboxes):
                # Randomise the location of the top left corner of the
                # bounding box
                x_min = random.randint(0, img_size - fixed_width_height)
                y_min = random.randint(0, img_size - fixed_width_height)

                # Add fixed width and height to get the bottom right corner
                x_max = x_min + fixed_width_height
                y_max = y_min + fixed_width_height
                boxes.append([x_min, y_min, x_max, y_max])
            annotations.append(torch.tensor(boxes))
        return images, annotations

    return _create_dummy_dataset


@pytest.fixture()
def create_dummy_dataset_dirs(create_dummy_dataset, tmp_path_factory):
    """Return a factory of dictionaries with dataset paths for testing.

    The linked datasets are N-images datasets with dummy annotations
    in COCO format.
    """

    def _create_dummy_dataset_dirs(n_images):
        # Get dummy data
        images, annotations = create_dummy_dataset(n_images)

        # Create temporary directories
        frames_dir = tmp_path_factory.mktemp("frames")
        annotations_dir = tmp_path_factory.mktemp("annotations")
        annotations_file_path = annotations_dir / "sample.json"

        # Save images to temporary directory
        for idx, img in enumerate(images):
            out_path = frames_dir / f"frame_{idx:04d}.png"
            save_image(img, out_path)

        # Save annotations file with expected format to temporary directory
        annotations_dict = bbox_tensors_to_COCO_dict(annotations)

        with open(annotations_file_path, "w") as f:
            json.dump(annotations_dict, f, indent=4)  # pretty print

        # Return paths as dict
        dataset_paths = {
            "frames": frames_dir,
            "annotations": annotations_file_path,
        }

        return dataset_paths

    return _create_dummy_dataset_dirs


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
    """Test transforms linked to training set are as expected."""
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
        # because it points to a lambda function, which does not have a
        # comparison defined.
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
    """Test transforms linked to test and validation sets are as expected."""
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
def test_collate_fn(crabs_data_module, create_dummy_dataset, request):
    """Test collate function formats the dataset as expected."""
    crabs_data_module = request.getfixturevalue(crabs_data_module)

    dataset = create_dummy_dataset(n_images=5)
    collated_data = crabs_data_module._collate_fn(dataset)

    assert len(collated_data) == len(dataset[0])  # images
    assert len(collated_data) == len(dataset[1])  # annotations

    for i, sample in enumerate(collated_data):
        # check length is 2 -> (image, annotation)
        assert len(sample) == 2

        # check content is the same as in input dataset
        image, annotation = sample
        assert torch.equal(image, dataset[0][i])
        assert torch.equal(annotation, dataset[1][i])


@pytest.mark.parametrize(
    (
        "dataset_size, seed, train_fraction, "
        "val_over_test_fraction, expected_img_ids_per_split"
    ),
    [
        (
            50,
            123,
            0.8,
            0.5,
            {"train": [33, 31, 1], "test": [21, 44, 41], "val": [36, 40, 27]},
        ),
        (
            100,
            42,
            0.6,
            0.5,
            {"train": [43, 97, 63], "test": [9, 66, 1], "val": [73, 91, 86]},
        ),
        (
            250,
            37,
            0.6,
            0.25,
            {
                "train": [32, 50, 119],
                "test": [107, 9, 68],
                "val": [199, 180, 168],
            },
        ),
    ],
)
def test_compute_splits(
    dataset_size,
    seed,
    train_fraction,
    val_over_test_fraction,
    expected_img_ids_per_split,
    create_dummy_dataset_dirs,
    default_train_config,
):
    """Test dataset splits are reproducible and match
    the requested fraction.
    """
    # Create a dummy dataset and get paths to its directories
    dataset_dirs = create_dummy_dataset_dirs(n_images=dataset_size)

    # Edit config to change splits' fractions
    default_train_config["train_fraction"] = train_fraction
    default_train_config["val_over_test_fraction"] = val_over_test_fraction

    # Create datamodule
    dm = CrabsDataModule(
        list_img_dirs=[dataset_dirs["frames"]],
        list_annotation_files=[dataset_dirs["annotations"]],
        config=default_train_config,
        split_seed=seed,
        no_data_augmentation=False,
    )

    # Compute splits
    train_transform = dm._get_test_val_transform()
    test_and_val_transform = dm._get_test_val_transform()
    train_dataset, _, _ = dm._compute_splits(train_transform)
    _, test_dataset, val_dataset = dm._compute_splits(test_and_val_transform)

    # Check total size of dataset
    total_dataset_size = (
        len(train_dataset) + len(test_dataset) + len(val_dataset)
    )
    n_frame_files = len(list(dataset_dirs["frames"].glob("*.png")))
    assert total_dataset_size == n_frame_files

    # Check split sizes match requested fractions
    assert np.isclose(
        len(train_dataset) / total_dataset_size, train_fraction, atol=0.05
    )
    assert np.isclose(
        len(test_dataset) / total_dataset_size,
        (1.0 - train_fraction) * (1.0 - val_over_test_fraction),
        atol=0.05,
    )
    assert np.isclose(
        len(val_dataset) / total_dataset_size,
        (1.0 - train_fraction) * val_over_test_fraction,
        atol=0.05,
    )

    # Check splits are non-overlapping in image IDs
    # Compute lists of image IDs per dataset
    image_ids_per_dataset = {}
    for dataset, dataset_str in zip(
        [train_dataset, test_dataset, val_dataset], ["train", "test", "val"]
    ):
        image_ids_per_dataset[dataset_str] = [
            sample[1]["image_id"] for sample in dataset
        ]

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

    # Check dataset creation is reproducible by checking
    # the first 3 image IDs are as expected
    assert (
        image_ids_per_dataset["train"][:3]
        == expected_img_ids_per_split["train"]
    )
    assert (
        image_ids_per_dataset["test"][:3] == expected_img_ids_per_split["test"]
    )
    assert (
        image_ids_per_dataset["val"][:3] == expected_img_ids_per_split["val"]
    )
