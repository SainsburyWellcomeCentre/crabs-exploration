from pathlib import Path

import pytest

from crabs.detection_tracking.train_model import train_parse_args


@pytest.mark.parametrize(
    "dataset_dirs_input",
    [["/home/data/dataset1"], ["/home/data/dataset1", "/home/data/dataset2"]],
)
def test_prep_img_directories(dataset_dirs_input: list):
    from crabs.detection_tracking.train_model import DectectorTrain

    # prepare parser
    train_args = train_parse_args(["--dataset_dirs"] + dataset_dirs_input)

    detector = DectectorTrain(train_args)
    detector.prep_img_directories(dataset_dirs_input)

    # assert
    list_imgs_dirs = [str(Path(d) / "frames") for d in dataset_dirs_input]
    assert detector.images_dirs == list_imgs_dirs


# # parametrise across different inputs for annotation_files
# # (None, single-path, single-filename, multiple-path, multiple-filename)
# def test_prep_annotation_files():
#     from crabs.detection_tracking.train_model import DectectorTrain

#     detector = DectectorTrain()
#     detector.prep_annotation_directories('crabs-exploration/tests/data/dataset')

#     # assert
#     detector.annotation_files
