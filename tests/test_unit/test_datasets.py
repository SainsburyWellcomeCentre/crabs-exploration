import random
from pathlib import Path

import pytest

from crabs.detection_tracking.datasets import CrabsCocoDetection

DATASET_1 = "/home/data/dataset1"
DATASET_2 = "/home/data/dataset2"

ANNOTATION_FILE = (
    Path(__file__).parents[1] / "data" / "sample_annotations_1.json"
)


@pytest.mark.parametrize("n_files_to_exclude", [0, 1, 20])
def test_exclude_files(n_files_to_exclude):
    # define list of annotation files
    list_datasets = [DATASET_1]
    list_annotations = [ANNOTATION_FILE]

    # create a dataset with all the images in the annotation file
    dataset = CrabsCocoDetection(
        list_datasets,
        list_annotations,
    )
    # check list of excluded files in dataset is None
    assert dataset.list_exclude_files is None

    # get all images in the combined dataset
    list_all_files = []
    for dataset in dataset.datasets:
        coco_object = dataset.coco
        list_files_in_dataset = [
            im["file_name"] for im in coco_object.dataset["images"]
        ]
        list_all_files.extend(list_files_in_dataset)

    # select a random subset of images in the dataset as images to exclude
    list_exclude_files = random.sample(list_all_files, n_files_to_exclude)

    # create a dataset from the same input data but without images to exclude
    dataset_w_exclude = CrabsCocoDetection(
        list_datasets,
        list_annotations,
        list_exclude_files=list_exclude_files,
    )

    # check number of samples in dataset with excluded images
    assert len(dataset) - n_files_to_exclude == len(dataset_w_exclude)

    # check list of excluded files
    assert dataset_w_exclude.list_exclude_files == list_exclude_files

    # check excluded images are not present in any of the concatenated datasets
    for dataset in dataset_w_exclude.datasets:
        coco_object = dataset.coco
        list_files_in_dataset = [
            im["file_name"] for im in coco_object.dataset["images"]
        ]
        assert all(
            [f not in list_files_in_dataset for f in list_exclude_files]
        )
