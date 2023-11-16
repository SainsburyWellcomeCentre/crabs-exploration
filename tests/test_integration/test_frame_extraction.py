from pathlib import Path

import pytest

from crabs.bboxes_labelling.extract_frames_to_label_w_sleap import \
    get_list_of_sleap_videos


@pytest.fixture(autouse=True, scope="class")
def input_video_dir():
    return Path(__file__).parent / "data" / "clips"


class TestsFrameExtraction:
    def test_extension_case_insensitive(self, input_video_dir):
        """
        Tests that the function that computes the list of SLEAP videos
        is case-insensitive for the user-input extension.

        Parameters
        ----------
        input_video_dir : pathlib.Path
            Path to the input video directory
        """
        # build list of video locations
        list_video_locations = [input_video_dir]

        # get unique extensions for all files in the
        # input directory
        # TODO: check they are all video files?
        list_files = [
            f
            for f in list_video_locations[0].glob("*")
            if f.is_file() and not f.name.startswith(".")
        ]
        list_unique_extensions = list({f.suffix[1:] for f in list_files})

        # force the user-input extensions to be of the opposite case
        list_user_extensions = [ext.lower() for ext in list_unique_extensions]
        list_user_extensions = list(set(list_user_extensions))

        # compute list of SLEAP videos for the given user extensions
        list_sleap_videos = get_list_of_sleap_videos(
            list_video_locations,
            list_user_extensions,
        )

        # check list of SLEAP videos matches the list of files
        assert len(list_sleap_videos) == len(list_files)
        assert len(list_sleap_videos) == len(list_files)
