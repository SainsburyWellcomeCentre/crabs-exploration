import datetime
import json
import os
import re
from pathlib import Path

import pytest
import typer
from typer.testing import CliRunner

from crabs.bboxes_labelling.extract_frames_to_label_w_sleap import (
    get_list_of_sleap_videos,
)

INPUT_DATA_DIR = str(Path(__file__).parents[1] / "data" / "clips")


def list_files_in_dir(input_dir: str) -> list:
    """Lists files in input directory

    Parameters
    ----------
    input_dir : str
        path to directory

    Returns
    -------
    list
        list of files in input directory
    """

    return [
        f
        for f in Path(input_dir).glob("*")
        if f.is_file() and not f.name.startswith(".")
    ]


def check_output_files(list_input_videos: list, cli_dict: dict) -> None:
    """Run assertions on output files from frame extraction

    Parameters
    ----------
    list_input_videos : list
        list of videos used for frame extraction
    cli_dict : dict
        a dictionary with the parameters of the frame extraction
    """
    # check name of directory with extracted frames
    list_subfolders = [
        f.path for f in os.scandir(cli_dict["output-path"]) if f.is_dir()
    ]
    extracted_frames_dir = Path(list_subfolders[0])
    assert len(list_subfolders) == 1
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

    if cli_dict["compute-features-per-video"]:
        n_expected_imgs = (
            cli_dict["n-clusters"]
            * cli_dict["per-cluster"]
            * len(list_input_videos)
        )
    else:
        n_expected_imgs = cli_dict["n-clusters"] * cli_dict["per-cluster"]

    assert len(list_imgs) <= n_expected_imgs

    # check filename format of images: <video_name>_frame_{frame_idx:08d}
    list_regex_patterns = [
        Path(input_video_str).stem + "_frame_[\d]{8}$"
        for input_video_str in list_input_videos
    ]
    for f in list_imgs:
        assert (
            sum(
                [
                    bool(re.fullmatch(regex, f.stem))
                    for regex in list_regex_patterns
                ]
            )
            == 1
        )  # only one must match

    # check n_elements in json file matches n of files generated
    with open((extracted_frames_dir / "extracted_frames.json")) as js:
        extracted_frames_dict = json.load(js)
        n_extracted_frames = sum(
            [len(list_idcs) for list_idcs in extracted_frames_dict.values()]
        )
        assert n_extracted_frames == len(list_imgs)


@pytest.fixture()
def video_extensions_flipped() -> list:
    """Extracts the extensions of video files in INPUT_DATA_DIR
    and flips their case (uppercase -> lowercase and viceversa).

    The file extensions would be provided by the user in the
    typical use case.
    """
    # build list of video files
    list_files = list_files_in_dir(INPUT_DATA_DIR)

    # get unique extensions for all files
    list_unique_extensions = list({f.suffix[1:] for f in list_files})

    # flip the case of the extensions
    list_extensions_flipped = [ext.lower() for ext in list_unique_extensions]
    list_extensions_flipped = list(set(list_extensions_flipped))

    return list_extensions_flipped


@pytest.mark.parametrize(
    "input_video",
    [
        "NINJAV_S001_S001_T003_subclip_p1_05s.mp4",
        "NINJAV_S001_S001_T003_subclip_p2_05s.MP4",
    ],
)
@pytest.mark.parametrize(
    "cli_inputs",
    [
        "cli_inputs_list",
        "cli_inputs_list_empty",
    ],
)
def test_frame_extraction_one_video(
    input_video: str,
    cli_inputs,
    cli_inputs_dict: dict,
    mock_extract_frames_app: typer.main.Typer,
    request,
) -> None:
    """Test frame extraction on one video, using default CLI arguments

    Parameters
    ----------
    input_video : str
        input video filename
    cli_inputs_dict : dict
        command line input arguments as a dictionary, for validation
    mock_extract_frames_app: typer.main.Typer
        a monkeypatched app with convenient defaults for testing
    """
    # import mocked app
    app = mock_extract_frames_app

    # prepare cli inputs
    cli_inputs_param = request.getfixturevalue(cli_inputs)

    # call mocked app
    runner = CliRunner()
    input_video_path = str(Path(INPUT_DATA_DIR) / input_video)
    result = runner.invoke(app, args=[input_video_path] + cli_inputs_param)
    assert result.exit_code == 0

    # check output files
    check_output_files([input_video_path], cli_inputs_dict)


@pytest.mark.parametrize(
    "cli_inputs",
    [
        "cli_inputs_list",
        "cli_inputs_list_empty",
    ],
)
def test_frame_extraction_one_dir(
    cli_inputs,
    cli_inputs_dict: dict,
    mock_extract_frames_app: typer.main.Typer,
    request,
) -> None:
    """Test frame extraction on one input directory, using default
    CLI arguments.

    Frames are extracted from all video files in the input
    directory.

    Parameters
    ----------
    cli_inputs_dict : dict
        command line input arguments as a dictionary, for validation
    mock_extract_frames_app : typer.main.Typer
        a monkeypatched app with convenient defaults for testing.
    """
    # import mock app
    app = mock_extract_frames_app

    # call cli inputs
    cli_inputs_list = request.getfixturevalue(cli_inputs)

    # invoke app
    runner = CliRunner()
    result = runner.invoke(app, args=[INPUT_DATA_DIR] + cli_inputs_list)

    # check exit code
    assert result.exit_code == 0

    # check files
    # list of input videos
    list_input_videos = list_files_in_dir(INPUT_DATA_DIR)
    list_input_videos = [
        f
        for f in list_input_videos
        if any(
            [
                str(f).lower().endswith(ext)
                for ext in cli_inputs_dict["video-extensions"]
            ]
        )
    ]
    check_output_files(list_input_videos, cli_inputs_dict)


def test_extension_case_insensitive(video_extensions_flipped: list) -> None:
    """
    Tests that the function that computes the list of SLEAP videos
    is case-insensitive for the user-provided extension.
    """

    # build list of video files in dir
    list_files = list_files_in_dir(INPUT_DATA_DIR)

    # compute list of SLEAP videos for the given user extensions;
    # the extensions are passed with the opposite case as the file extensions
    list_sleap_videos = get_list_of_sleap_videos(
        [INPUT_DATA_DIR],
        video_extensions_flipped,
    )

    # check list of SLEAP videos matches the list of files
    assert len(list_sleap_videos) == len(list_files)
    assert len(list_sleap_videos) == len(list_files)
