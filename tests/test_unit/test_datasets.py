from pathlib import Path

import pytest
from pycocotools.coco import COCO

from crabs.detection_tracking.datasets import CrabsCocoDetection
from crabs.detection_tracking.train_model import DEFAULT_ANNOTATIONS_FILENAME

DATASET_DIR = Path(__file__).parents[1] / "data" / "dataset"


@pytest.mark.parametrize("n_files_to_exclude", [0, 1, 20])
def test_exclude_files_single_dataset(n_files_to_exclude):
    # define reference annotation file
    annotations_file_path = str(
        Path(DATASET_DIR) / "annotations" / DEFAULT_ANNOTATIONS_FILENAME
    )

    # create a dataset with all the images in the annotation file
    dataset = CrabsCocoDetection(
        [DATASET_DIR],
        [annotations_file_path],
    )
    # check list of excluded files
    assert dataset.list_exclude_files is None

    # select a subset of images in the dataset to exclude
    coco = COCO(annotations_file_path)
    list_exclude_files = [
        coco.loadImgs(idx)[0]["file_name"]
        for idx in range(1, n_files_to_exclude + 1)
    ]

    # create a dataset from the same input data but without images to exclude
    dataset_w_exclude = CrabsCocoDetection(
        [DATASET_DIR],
        [annotations_file_path],
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
            im["file_name"] for im_id, im in coco_object.imgs.items()
        ]
        assert all(
            [f not in list_files_in_dataset for f in list_exclude_files]
        )
