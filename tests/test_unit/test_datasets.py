import glob
import mmap
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

    # exclude images from the same input data and check number of samples
    dataset_w_exclude = CrabsCocoDetection(
        [SAMPLE_DATASET_DIR],
        [annotation_file],
        list_exclude_files=list_exclude_files,
    )
    assert len(dataset) - n_files_to_exclude == len(dataset_w_exclude)

    # get list of output annotation files and check length
    glob_pattern = str(
        Path(annotation_file).parent / (Path(annotation_file).stem + "*")
    )
    list_output_files = glob.glob(glob_pattern)
    assert len(list_output_files) == 2

    # check new annotation file contains no references to excluded files
    new_output_file = [f for f in list_output_files if "_filt_" in f][0]
    with open(new_output_file) as file:
        s = mmap.mmap(file.fileno(), 0, access=mmap.ACCESS_READ)
        for img_to_excl in list_exclude_files:
            assert s.find(bytes(img_to_excl, "utf-8")) == -1
