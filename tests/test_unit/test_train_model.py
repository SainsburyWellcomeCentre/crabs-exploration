from pathlib import Path

import pytest

from crabs.detector.train_model import train_parse_args
from crabs.detector.utils.detection import (
    DEFAULT_ANNOTATIONS_FILENAME as DEFAULT_ANNOT_FILENAME,
)

DATASET_1 = "/home/data/dataset1"
DATASET_2 = "/home/data/dataset2"
ANNOTATION_FILE_1 = str(Path(DATASET_1) / "annotations" / "annotations1.json")
ANNOTATION_FILE_2 = str(Path(DATASET_2) / "annotations" / "annotations2.json")


@pytest.mark.parametrize(
    "dataset_dirs",
    [["/home/data/dataset1"], ["/home/data/dataset1", "/home/data/dataset2"]],
)
def test_prep_img_directories(dataset_dirs: list):
    """Test parsing of image directories when training a model."""
    from crabs.detector.train_model import DetectorTrain

    # prepare parser
    train_args = train_parse_args(["--dataset_dirs"] + dataset_dirs)

    # instantiate detector
    detector = DetectorTrain(train_args)

    # check image directories are parsed correctly
    list_imgs_dirs = [str(Path(d) / "frames") for d in dataset_dirs]
    assert detector.images_dirs == list_imgs_dirs


@pytest.mark.parametrize(
    "annotation_files, expected",
    [
        (
            [],
            [str(Path(DATASET_1) / "annotations" / DEFAULT_ANNOT_FILENAME)],
        ),  # default input
        (["annotations1.json"], [ANNOTATION_FILE_1]),  # filename only
        ([ANNOTATION_FILE_2], [ANNOTATION_FILE_2]),  # fullpath
    ],
)
def test_prep_annotation_files_single_dataset(annotation_files, expected):
    """Test parsing of annotation files when training a model on a single
    dataset.
    """
    from crabs.detector.train_model import DetectorTrain

    # prepare CLI arguments
    cli_inputs = ["--dataset_dirs", DATASET_1]

    # if annotation_files is not an empty list:
    # append annotations file
    if annotation_files:
        cli_inputs.append("--annotation_files")
    train_args = train_parse_args(cli_inputs + annotation_files)

    # instantiate detector
    detector = DetectorTrain(train_args)

    # check annotation files are as expected
    assert detector.annotation_files == expected


@pytest.mark.parametrize(
    "annotation_files, expected",
    [
        (
            [],
            [
                str(Path(dataset) / "annotations" / DEFAULT_ANNOT_FILENAME)
                for dataset in [DATASET_1, DATASET_2]
            ],
        ),  # default input
        (
            ["annotations1.json", "annotations2.json"],
            [ANNOTATION_FILE_1, ANNOTATION_FILE_2],
        ),  # filename only
        (
            [ANNOTATION_FILE_1, ANNOTATION_FILE_2],
            [ANNOTATION_FILE_1, ANNOTATION_FILE_2],
        ),  # fullpath
    ],
)
def test_prep_annotation_files_multiple_datasets(annotation_files, expected):
    """Test parsing of annotation files when training
    a model on two datasets.
    """
    from crabs.detector.train_model import DetectorTrain

    # prepare CLI arguments considering multiple dataset
    cli_inputs = ["--dataset_dirs", DATASET_1, DATASET_2]

    # if  annotation_files_input is not an empty list:
    # append annotations file
    if annotation_files:
        cli_inputs.append("--annotation_files")
    train_args = train_parse_args(cli_inputs + annotation_files)

    # instantiate detector
    detector = DetectorTrain(train_args)

    # check annotation files are as expected
    assert detector.annotation_files == expected
