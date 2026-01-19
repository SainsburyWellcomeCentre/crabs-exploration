from pathlib import Path

import pytest

from crabs.detector.utils.detection import (
    DEFAULT_ANNOTATIONS_FILENAME,
    prep_annotation_files,
    prep_img_directories,
)

DATASET_1 = "/home/data/dataset1"
DATASET_2 = "/home/data/dataset2"


@pytest.mark.parametrize(
    "input_datasets",
    [
        [DATASET_1],
        [DATASET_1, DATASET_2],
    ],
)
def test_prep_img_directories(input_datasets):
    expected_img_dirs = []
    for dataset_dir in input_datasets:
        expected_img_dirs.append(str(Path(dataset_dir) / "frames"))

    assert prep_img_directories(input_datasets) == expected_img_dirs


@pytest.mark.parametrize(
    "input_datasets",
    [
        [DATASET_1],
        [DATASET_1, DATASET_2],
    ],
)
def test_prep_annotation_files_default(input_datasets):
    expected_annot_paths = []
    for dataset_dir in input_datasets:
        expected_annot_paths.append(
            str(
                Path(dataset_dir)
                / "annotations"
                / DEFAULT_ANNOTATIONS_FILENAME
            )
        )

    assert prep_annotation_files([], input_datasets) == expected_annot_paths


@pytest.mark.parametrize(
    "input_datasets, input_annotation_files",
    [
        ([DATASET_1], ["file1.json"]),
        ([DATASET_1, DATASET_2], ["file1.json", "file2.json"]),
    ],
)
def test_prep_annotation_files_with_filenames(
    input_datasets, input_annotation_files
):
    expected_annot_paths = []
    for dataset_dir, annot_file in zip(
        input_datasets, input_annotation_files, strict=False
    ):
        expected_annot_paths.append(
            str(Path(dataset_dir) / "annotations" / annot_file)
        )

    assert (
        prep_annotation_files(input_annotation_files, input_datasets)
        == expected_annot_paths
    )


@pytest.mark.parametrize(
    "input_datasets, input_annotation_files",
    [
        ([DATASET_1], ["path/to/file1.json"]),
        ([DATASET_1, DATASET_2], ["path/to/file1.json", "path/to/file2.json"]),
    ],
)
def test_prep_annotation_files_with_fullpaths(
    input_datasets, input_annotation_files
):
    expected_annot_paths = []
    for annot_file in input_annotation_files:
        expected_annot_paths.append(annot_file)

    assert (
        prep_annotation_files(input_annotation_files, input_datasets)
        == expected_annot_paths
    )
