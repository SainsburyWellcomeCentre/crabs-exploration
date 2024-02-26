from pathlib import Path

import pytest

from crabs.bboxes_labelling.extract_frames_to_label_w_sleap import (
    get_list_of_sleap_videos,
)


@pytest.fixture(autouse=True, scope="class")
def input_video_dir():
    return Path(__file__).parents[1] / "data" / "clips"


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
