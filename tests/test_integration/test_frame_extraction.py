import subprocess
from pathlib import Path

import pytest

from crabs.bboxes_labelling.extract_frames_to_label_w_sleap import (
    get_list_of_sleap_videos,
)


@pytest.fixture()
def extract_frames_command() -> str:
    return "extract-frames"


@pytest.fixture()
def input_data_dir() -> str:
    return str(Path(__file__).parents[1] / "data" / "clips")


@pytest.fixture(
    params=[
        "NINJAV_S001_S001_T003_subclip_p1_05s.mp4",
        "NINJAV_S001_S001_T003_subclip_p2_05s.mp4",
    ]
)
def input_video(input_data_dir, request) -> str:
    return str(Path(input_data_dir) / request.param)


@pytest.fixture()
def cli_input_arguments(tmp_path: Path) -> dict:
    return {
        "output-path": str(tmp_path),
        "video-extensions": "mp4",
        "initial-samples": "5",
        "scale": "0.5",
        "n-components": "3",
        "n-clusters": "5",
        "per-cluster": "1",
        "compute-features-per-video": "",
    }


@pytest.fixture()
def video_extensions_flipped(input_data_dir: str) -> list:
    # build list of video files
    list_files = list_files_in_dir(input_data_dir)

    # get unique extensions for all files
    list_unique_extensions = list({f.suffix[1:] for f in list_files})

    # flip the case of the extensions
    list_user_extensions_flip = [ext.lower() for ext in list_unique_extensions]
    list_user_extensions_flip = list(set(list_user_extensions_flip))

    return list_user_extensions_flip


def dict_to_list_of_cli_args(input_params: dict) -> list:
    """Transforms a dictionary of parameters into a list of CLI arguments
    that can be passed to `subprocess.run()`.

    If for an item in the dictionary the value is empty string,
    its key is taken as a CLI boolean argument (i.e., a flag).

    Parameters
    ----------
    input_params : dict
        dictionary with the command line arguments to transform.
        If a value is an empty string, the corresponding key is
        considered a boolean input argument (i.e., a flag).

    Returns
    -------
    list
        a list of command line arguments to pass to `subprocess.run()`.
    """

    list_kys_modified = ["--" + k for k in input_params.keys()]
    list_cli_args = [
        elem
        for pair in zip(list_kys_modified, input_params.values())
        for elem in pair
        if elem != ""
    ]

    return list_cli_args


def list_files_in_dir(input_dir: str):
    # build list of files in dir
    return [
        f
        for f in Path(input_dir).glob("*")
        if f.is_file() and not f.name.startswith(".")
    ]


def test_small_frame_extraction_one_video(
    extract_frames_command: str,
    input_video: str,
    cli_input_arguments: dict,
):
    # format input parameters as command line arguments
    list_cli_args = dict_to_list_of_cli_args(cli_input_arguments)

    # run extract frames on one video
    result = subprocess.run(
        [
            extract_frames_command,
            input_video,
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
    extract_frames_command: str,
    input_data_dir: str,
    cli_input_arguments: dict,
):
    # format input parameters as command line arguments
    list_cli_args = dict_to_list_of_cli_args(cli_input_arguments)

    result = subprocess.run(
        [
            extract_frames_command,
            input_data_dir,
        ]
        + list_cli_args,  # comment for defaults?
        capture_output=True,
        text=True,
    )

    # check return code
    assert result.returncode == 0

    # check name of files


def test_extension_case_insensitive(
    input_data_dir: str, video_extensions_flipped: list
):
    """
    Tests that the function that computes the list of SLEAP videos
    is case-insensitive for the user-input extension.

    Parameters
    ----------
    input_video_dir : pathlib.Path
        Path to the input video directory
    """

    # build list of video files in dir
    list_files = list_files_in_dir(input_data_dir)

    # compute list of SLEAP videos for the given user extensions;
    # the extensions are passed with the opposite case as the file extensions
    list_sleap_videos = get_list_of_sleap_videos(
        [input_data_dir],
        video_extensions_flipped,
    )

    # check list of SLEAP videos matches the list of files
    assert len(list_sleap_videos) == len(list_files)
    assert len(list_sleap_videos) == len(list_files)
