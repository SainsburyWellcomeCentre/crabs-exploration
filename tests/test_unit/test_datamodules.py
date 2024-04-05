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


def create_dummy_dataset(num_samples):
    """Create dummy images and annotations for testing."""
    images = [torch.randn(3, 256, 256) for _ in range(num_samples)]
    annotations = [
        torch.randint(0, 10, size=(4, 5)) for _ in range(num_samples)
    ]
    return (images, annotations)


def test_collate_fn(crabs_data_module):
    num_samples = 5
    sample_input = create_dummy_dataset(num_samples)

    collated_data = crabs_data_module._collate_fn(sample_input)

    for sample in collated_data:
        image, annotation = sample

        assert isinstance(
            image, torch.Tensor
        ), "Image should be a torch.Tensor"
        assert isinstance(
            annotation, torch.Tensor
        ), "Annotation should be a torch.Tensor"


# def test_compute_splits():
#     list_img_dirs = ["/Users/nikkhadijahnikaznan/Git/crabs-exploration/tests/data/sample_data/frames"]
#     list_annotation_files = ["/Users/nikkhadijahnikaznan/Git/crabs-exploration/tests/data/sample_data/annotations/sample_annotation_coco.json"]
#     print(list_img_dirs)
#     print(list_annotation_files)
#     config = {"train_fraction": 1.0, "val_over_test_fraction": 0.0}
#     crabs_data_module = CrabsDataModule(
#         list_img_dirs,
#         list_annotation_files,
#         config,
#         22,
#     )

#     train_dataset, test_dataset, val_dataset = crabs_data_module._compute_splits(transforms_all_splits=None)

#     assert len(train_dataset) == 2
#     assert len(test_dataset) == 1
#     assert len(val_dataset) == 1
