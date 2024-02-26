import subprocess
from pathlib import Path

import pytest

from crabs.bboxes_labelling.extract_frames_to_label_w_sleap import (
    get_list_of_sleap_videos,
)


@pytest.fixture(autouse=True, scope="class")
def input_video_dir():
    return Path(__file__).parents[1] / "data" / "clips"


@pytest.fixture()
def extract_frames_command():
    return "extract-frames"


@pytest.fixture()
def sample_input_video():
    # TODO: add a very small sample clip!
    return Path(__file__).parent / "data" / "NINJAV_S001_S001_T003_subclip.mp4"


@pytest.fixture()
def sample_input_dir():
    return Path(__file__).parent / "data"


@pytest.fixture()
def sample_output_dir():
    return Path(__file__).parent / "output"


@pytest.fixture()
def sample_input_params(sample_output_dir):
    return {
        "output_path": sample_output_dir,
        "video_extensions": "mp4",
        "initial_samples": "5",
        "scale": "0.5",
        "n_components": "3",
        "n_clusters": "5",
        "per_cluster": "1",
    }


@pytest.fixture()
def list_user_extensions_flipped(input_video_dir):
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


def test_small_frame_extraction_one_video(
    extract_frames_command, sample_input_video, sample_input_params
):
    # format input parameters as command line arguments
    list_kys_modif = ["--" + k for k in sample_input_params.keys()]
    list_non_bool_cli_args = [
        val
        for pair in zip(list_kys_modif, list(sample_input_params.values()))
        for val in pair
    ]
    list_bool_cli_args = ["--compute_features_per_video"]
    list_cli_args = list_non_bool_cli_args + list_bool_cli_args

    result = subprocess.run(
        [
            extract_frames_command,
            sample_input_video,
        ]
        + list_cli_args,
        capture_output=True,
        text=True,
    )

    # check return code
    assert result.returncode == 0

    # check one json file

    # check n_elements in json file matches n of files generated

    # check min number of files? (NOTE: total number of files is not deterministic!)

    # check name of files


def test_small_frame_extraction_one_dir(
    extract_frames_command, sample_input_params, sample_input_dir
):
    # TODO: can these be fixtures?
    list_kys_modif = ["--" + k for k in sample_input_params.keys()]
    list_non_bool_cli_args = [
        val
        for pair in zip(list_kys_modif, list(sample_input_params.values()))
        for val in pair
    ]
    list_bool_cli_args = ["--compute_features_per_video"]
    list_cli_args = list_non_bool_cli_args + list_bool_cli_args

    result = subprocess.run(
        [
            extract_frames_command,
            sample_input_dir,
        ]
        + list_cli_args,
        capture_output=True,
        text=True,
    )

    # check return code
    assert result.returncode == 0

    # check name of files


def test_extension_case_insensitive(
    input_video_dir, list_user_extensions_flipped
):
    """
    Tests that the function that computes the list of SLEAP videos
    is case-insensitive for the user-input extension.

    Parameters
    ----------
    input_video_dir : pathlib.Path
        Path to the input video directory
    """
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
