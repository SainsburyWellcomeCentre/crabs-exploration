import subprocess
from pathlib import Path

import pytest

from crabs.bboxes_labelling.extract_frames_to_label_w_sleap import (
    get_list_of_sleap_videos,
)

# @pytest.fixture(autouse=True, scope="class")
# def input_video_dir():
#     return Path(__file__).parents[1] / "data" / "clips"


@pytest.fixture()
def extract_frames_command():
    return "extract-frames"


@pytest.fixture()
def sample_data(tmp_path):
    input_dir = Path(__file__).parents[1] / "data" / "clips"
    output_dir = tmp_path

    return {
        "input_dir": input_dir,
        "output_dir": output_dir,
        "input_video": input_dir / "NINJAV_S001_S001_T003_subclip.mp4",
        "extract_frames_params": {
            "output_path": str(output_dir),
            "video_extensions": "mp4",
            "initial_samples": "5",
            "scale": "0.5",
            "n_components": "3",
            "n_clusters": "5",
            "per_cluster": "1",
            "compute_features_per_video": "",
        },
    }


def dict_to_list_of_cli_args(input_params: dict) -> list:
    """If value is empty string, key is taken as a CLI boolean argument

    Parameters
    ----------
    input_params : dict
        _description_

    Returns
    -------
    list
        _description_
    """

    list_kys_modified = ["--" + k for k in input_params.keys()]
    list_cli_args = [
        elem
        for pair in zip(list_kys_modified, input_params.values())
        for elem in pair
        if elem != ""
    ]

    return list_cli_args


@pytest.fixture()
def list_user_extensions_flipped(sample_data):
    input_video_dir = sample_data["input_dir"]

    # build list of video files
    list_files = [
        f
        for f in input_video_dir.glob("*")
        if f.is_file() and not f.name.startswith(".")
    ]

    # get unique extensions for all files in the
    # input directory
    list_unique_extensions = list({f.suffix[1:] for f in list_files})

    # force the user-input extensions to be of the opposite case
    list_user_extensions_flip = [ext.lower() for ext in list_unique_extensions]
    list_user_extensions_flip = list(set(list_user_extensions_flip))

    return list_user_extensions_flip


def test_small_frame_extraction_one_video(extract_frames_command, sample_data):
    # format input parameters as command line arguments
    dict_to_list_of_cli_args(sample_data["extract_frames_params"])

    result = subprocess.run(
        [
            extract_frames_command,
            str(sample_data["input_video"]),
        ],
        # + list_cli_args,
        capture_output=True,
        text=True,
    )

    # check return code
    assert result.returncode == 0

    # check one json file

    # check n_elements in json file matches n of files generated

    # check min number of files? (NOTE: total number of files is not deterministic!)

    # check name of files


def test_small_frame_extraction_one_dir(extract_frames_command, sample_data):
    # format input parameters as command line arguments
    dict_to_list_of_cli_args(sample_data["extract_frames_params"])

    result = subprocess.run(
        [
            extract_frames_command,
            str(sample_data["input_dir"]),
        ],
        # + list_cli_args,  # comment for defaults?
        capture_output=True,
        text=True,
    )

    # check return code
    assert result.returncode == 0

    # check name of files


def test_extension_case_insensitive(sample_data, list_user_extensions_flipped):
    """
    Tests that the function that computes the list of SLEAP videos
    is case-insensitive for the user-input extension.

    Parameters
    ----------
    input_video_dir : pathlib.Path
        Path to the input video directory
    """

    input_video_dir = sample_data["input_dir"]

    # build list of video files
    list_files = [
        f
        for f in input_video_dir.glob("*")
        if f.is_file() and not f.name.startswith(".")
    ]

    # compute list of SLEAP videos for the given user extensions
    list_sleap_videos = get_list_of_sleap_videos(
        [input_video_dir],
        list_user_extensions_flipped,
    )

    # check list of SLEAP videos matches the list of files
    assert len(list_sleap_videos) == len(list_files)
    assert len(list_sleap_videos) == len(list_files)
