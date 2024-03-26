import random
from pathlib import Path

import pytest

from crabs.detection_tracking.datasets import CrabsCocoDetection

DATASET_1 = "/home/data/dataset1"
DATASET_2 = "/home/data/dataset2"

ANNOTATION_FILE_1 = (
    Path(__file__).parents[1]
    / "data"
    / "annotations"
    / "sample_annotations_1.json"
)  # 100 images, 4618 annotations

ANNOTATION_FILE_2 = (
    Path(__file__).parents[1]
    / "data"
    / "annotations"
    / "sample_annotations_2.json"
)  # 100 images, 4344 annotations


@pytest.mark.parametrize(
    "list_datasets, list_annotations",
    [
        ([DATASET_1], [ANNOTATION_FILE_1]),
        ([DATASET_1, DATASET_2], [ANNOTATION_FILE_1, ANNOTATION_FILE_2]),
    ],
)
@pytest.mark.parametrize("n_files_to_exclude", [0, 1, 20, 97])
def test_exclude_files(list_datasets, list_annotations, n_files_to_exclude):
    """
    Test if the required files are excluded correctly from the
    dataset defined by a list of annotation files, and a list of
    corresponding image directories.

    Note that n_files_to_exclude should be lower than the full dataset size.

    """
    # Create a dataset with all the input data
    full_dataset = CrabsCocoDetection(
        list_datasets,
        list_annotations,
    )
    # Check list of excluded files in dataset is None
    assert full_dataset.list_exclude_files is None

    # Select a subset of images in the full dataset as images to exclude.
    # If multiple dataset: sample images from each dataset if possible.
    n_datasets = len(list_datasets)
    q, r = divmod(n_files_to_exclude, n_datasets)
    # If n_files_to_exclude > n_datasets: add remained to last dataset
    if q >= 1:
        n_exclude_per_dataset = [q] * n_datasets
        n_exclude_per_dataset[-1] += r
    # If n_files_to_exclude < n_datasets:
    # sample one random image from `n_files_to_exclude` datasets
    else:
        n_exclude_per_dataset = [1] * r + [0] * (n_datasets - r)
        random.shuffle(n_exclude_per_dataset)

    # Determine images to exclude
    list_exclude_files = []
    for dataset, n_excl in zip(full_dataset.datasets, n_exclude_per_dataset):
        coco_object = dataset.coco
        list_files_in_dataset = [
            im["file_name"] for im in coco_object.dataset["images"]
        ]

        list_exclude_files.extend(random.sample(list_files_in_dataset, n_excl))

    # Create a dataset from the same input data but without images to exclude
    dataset_w_exclude = CrabsCocoDetection(
        list_datasets,
        list_annotations,
        list_exclude_files=list_exclude_files,
    )

    # Check number of samples in dataset with excluded images
    assert len(full_dataset) - n_files_to_exclude == len(dataset_w_exclude)

    # Check list of excluded files
    assert dataset_w_exclude.list_exclude_files == list_exclude_files

    # Check excluded images are not present in any of the concatenated datasets
    for dataset in dataset_w_exclude.datasets:
        coco_object = dataset.coco
        list_files_in_dataset = [
            im["file_name"] for im in coco_object.dataset["images"]
        ]
        assert all(
            [f not in list_files_in_dataset for f in list_exclude_files]
        )
