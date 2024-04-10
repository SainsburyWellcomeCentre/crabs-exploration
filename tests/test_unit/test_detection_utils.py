from pathlib import Path

import pytest

from crabs.detection_tracking.detection_utils import (
    prep_annotation_files,
    prep_img_directories,
)


@pytest.fixture
def dataset_dirs(tmp_path):
    dataset1_dir = tmp_path / "dataset1"
    dataset1_dir.mkdir()
    (dataset1_dir / "frames").mkdir()

    dataset2_dir = tmp_path / "dataset2"
    dataset2_dir.mkdir()
    (dataset2_dir / "frames").mkdir()

    return [str(dataset1_dir), str(dataset2_dir)]


def test_prep_img_directories(dataset_dirs):
    expected_result = [
        str(Path(dataset_dirs[0]) / "frames"),
        str(Path(dataset_dirs[1]) / "frames"),
    ]
    assert prep_img_directories(dataset_dirs) == expected_result


def test_prep_annotation_files_default(dataset_dirs):
    result = prep_annotation_files([], dataset_dirs)
    print(result)
    expected_result = [
        str(
            Path(dataset_dirs[0])
            / "annotations"
            / "VIA_JSON_combined_coco_gen.json"
        ),
        str(
            Path(dataset_dirs[1])
            / "annotations"
            / "VIA_JSON_combined_coco_gen.json"
        ),
    ]
    assert result == expected_result


def test_prep_annotation_files_with_filenames(dataset_dirs):
    input_annotation_files = ["file1.json", "file2.json"]
    result = prep_annotation_files(input_annotation_files, dataset_dirs)
    expected_result = [
        str(Path(dataset_dirs[0]) / "annotations" / "file1.json"),
        str(Path(dataset_dirs[1]) / "annotations" / "file2.json"),
    ]
    assert result == expected_result


def test_prep_annotation_files_with_full_paths(dataset_dirs):
    input_annotation_files = ["/path/to/file1.json", "/path/to/file2.json"]
    result = prep_annotation_files(input_annotation_files, dataset_dirs)
    assert result == input_annotation_files
