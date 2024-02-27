import datetime
import os
import re
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
def cli_dict(tmp_path: Path) -> dict:
    """These are also the default values

    Parameters
    ----------
    tmp_path : Path
        _description_

    Returns
    -------
    dict
        _description_
    """
    return {
        "output-path": str(tmp_path),
        "video-extensions": "mp4",
        "initial-samples": 5,
        "scale": 0.5,
        "n-components": 3,
        "n-clusters": 5,
        "per-cluster": 1,
        "compute-features-per-video": "",
    }


@pytest.fixture()
def cli_input_arguments(cli_dict: dict) -> list:
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
            str(elem)
            for pair in zip(list_kys_modified, input_params.values())
            for elem in pair
            if elem != ""
        ]

        return list_cli_args

    return dict_to_list_of_cli_args(cli_dict)


@pytest.fixture()
def cli_default_arguments() -> list:
    return []


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


def list_files_in_dir(input_dir: str):
    # build list of files in dir
    return [
        f
        for f in Path(input_dir).glob("*")
        if f.is_file() and not f.name.startswith(".")
    ]


def check_output_files(input_video_str, cli_dict):
    # check name of directory with extracted frames
    list_subfolders_with_paths = [
        f.path for f in os.scandir(cli_dict["output-path"]) if f.is_dir()
    ]
    extracted_frames_dir = Path(list_subfolders_with_paths[0])

    assert len(list_subfolders_with_paths) == 1
    assert (
        type(
            datetime.datetime.strptime(
                extracted_frames_dir.name, "%Y%m%d_%H%M%S"
            )
        )
        == datetime.datetime
    )

    # check there is an extracted_frames.json file
    assert (extracted_frames_dir / "extracted_frames.json").is_file()

    # check min number of image files
    # NOTE: total number of files generated is not deterministic
    list_imgs = [f for f in extracted_frames_dir.glob("*.png")]
    assert len(list_imgs) <= cli_dict["n-clusters"] * cli_dict["per-cluster"]

    # check format of images filename
    regex_pattern = Path(input_video_str).stem + "_frame_[\d]{8}$"
    assert all([re.fullmatch(regex_pattern, f.stem) for f in list_imgs])

    # check n_elements in json file matches n of files generated


@pytest.mark.parametrize(
    "cli_input",
    ["cli_input_arguments"],  # , "cli_default_arguments"]
)
def test_small_frame_extraction_one_video(
    extract_frames_command: str,
    input_video: str,
    cli_input: list,
    cli_dict: dict,  # ---- make these hold the default values!
    request: pytest.FixtureRequest,
    tmp_path: Path,
):
    # get CLI arguments fixture
    list_cli_args = request.getfixturevalue(cli_input)

    # run extract frames on one video from pytest tmp path
    result = subprocess.run(
        [
            extract_frames_command,
            input_video,
        ]
        + list_cli_args,
        capture_output=True,
        text=True,
    )

    # check return code of subprocess
    assert result.returncode == 0

    # check output files
    check_output_files(input_video, cli_dict)


@pytest.mark.parametrize(
    "cli_input", ["cli_input_arguments", "cli_default_arguments"]
)
def test_small_frame_extraction_one_dir(
    extract_frames_command: str,
    input_data_dir: str,
    cli_input: list,
    request: pytest.FixtureRequest,
):
    # get fixture for CLI input arguments
    list_cli_args = request.getfixturevalue(cli_input)

    # run command
    result = subprocess.run(
        [
            extract_frames_command,
            input_data_dir,
        ]
        + list_cli_args,
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
