import shutil
from pathlib import Path

import pytest
from pycocotools.coco import COCO

from crabs.detection_tracking.datasets import CrabsCocoDetection
from crabs.detection_tracking.train_model import DEFAULT_ANNOTATIONS_FILENAME

SAMPLE_DATASET_DIR = Path(__file__).parents[1] / "data" / "dataset"


@pytest.fixture()
def annotation_file(tmp_path: Path):
    """Copy sample annotation file to tmp_path"""

    src_annotation_file = (
        SAMPLE_DATASET_DIR / "annotations" / DEFAULT_ANNOTATIONS_FILENAME
    )
    out_annotation_file = tmp_path / DEFAULT_ANNOTATIONS_FILENAME
    shutil.copyfile(src_annotation_file, out_annotation_file)

    return str(out_annotation_file)


@pytest.mark.parametrize("n_files_to_exclude", [1, 20])
def test_exclude_files(annotation_file, n_files_to_exclude):
    # create dataset with all images
    dataset = CrabsCocoDetection(
        [SAMPLE_DATASET_DIR],
        [annotation_file],
    )

    # select a subset of images in the dataset
    coco = COCO(annotation_file)
    list_exclude_files = [
        coco.loadImgs(idx)[0]["file_name"]
        for idx in range(1, n_files_to_exclude + 1)
    ]

    # exclude images and check number of samples
    dataset_w_exclude = CrabsCocoDetection(
        [SAMPLE_DATASET_DIR],
        [annotation_file],
        list_exclude_files=list_exclude_files,
    )
    assert len(dataset) - n_files_to_exclude == len(dataset_w_exclude)

    # # check excluded files not in new dataset
    # coco_object = dataset_w_exclude._coco_obj_per_dataset[str(SAMPLE_DATASET_DIR)]
    # filenames_in_dataset_w_exclude = [
    #     val["file_name"] for _, val in coco_object.imgs.items()
    # ]
    # assert all(
    #     [f not in filenames_in_dataset_w_exclude for f in list_exclude_files]
    # )
