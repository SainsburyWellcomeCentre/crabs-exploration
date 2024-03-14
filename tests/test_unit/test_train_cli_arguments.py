from pathlib import Path

import pytest

from crabs.detection_tracking.train_model import (
    DEFAULT_ANNOTATIONS_FILE_REL,
    train_parse_args,
)

SAMPLE_DATASET_DIR = "/home/data/dataset1"
SAMPLE_ANNOTATION_FILE_1 = str(
    Path(SAMPLE_DATASET_DIR) / "annotations" / "annotations1.json"
)
SAMPLE_ANNOTATION_FILE_2 = "/home/data/dataset2/annotations/annotations2.json"


@pytest.mark.parametrize(
    "dataset_dirs_input",
    [["/home/data/dataset1"], ["/home/data/dataset1", "/home/data/dataset2"]],
)
def test_prep_img_directories(dataset_dirs_input: list):
    from crabs.detection_tracking.train_model import DectectorTrain

    # prepare parser
    train_args = train_parse_args(["--dataset_dirs"] + dataset_dirs_input)

    # instantiate detector
    detector = DectectorTrain(train_args)

    # check image directories are parsed correctly
    list_imgs_dirs = [str(Path(d) / "frames") for d in dataset_dirs_input]
    assert detector.images_dirs == list_imgs_dirs


# parametrise across different inputs for annotation_files
# # (None, single-path, single-filename, multiple-path, multiple-filename)
@pytest.mark.parametrize(
    "annotation_files_input,expected",
    [
        ([], [str(Path(SAMPLE_DATASET_DIR) / DEFAULT_ANNOTATIONS_FILE_REL)]),
        (["annotations1.json"], [SAMPLE_ANNOTATION_FILE_1]),
        ([SAMPLE_ANNOTATION_FILE_2], [SAMPLE_ANNOTATION_FILE_2]),
    ],
)
def test_prep_annotation_files(annotation_files_input, expected):
    from crabs.detection_tracking.train_model import DectectorTrain

    # prepare parser
    cli_inputs = ["--dataset_dirs", SAMPLE_DATASET_DIR]
    if annotation_files_input:
        cli_inputs.append("--annotation_files")
    train_args = train_parse_args(cli_inputs + annotation_files_input)

    detector = DectectorTrain(train_args)

    assert detector.annotation_files == expected
