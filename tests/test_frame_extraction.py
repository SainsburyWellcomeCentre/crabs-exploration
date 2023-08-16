from pathlib import Path
from bboxes_labelling.extract_frames_to_label_w_sleap import get_list_of_sleap_videos
import pytest


@pytest.fixture(autouse=True, scope="class")
def input_video_dir():
    return Path(__file__).parent / "data"


class TestsFrameExtraction:
    def test_extension_case_insensitive(self, input_video_dir):
        """
        Tests that the function that computes the list of SLEAP videos
        is case-insensitive for the user-input extension

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
            for f in list_video_locations[0].glob("**/*")
            if f.is_file() and not f.name.startswith(".")
        ]
        list_unique_extensions = list(set([f.suffix[1:] for f in list_files]))

        # force the user-input extension to be of the opposite case
        list_video_extensions = []
        for ext in list_unique_extensions:
            if ext.isupper():
                list_video_extensions.append(ext.lower())
            elif ext.islower():
                list_video_extensions.append(ext.upper())

        # compute list of SLEAP videos
        list_sleap_videos = get_list_of_sleap_videos(
            list_video_locations, list_video_extensions
        )

        # check list of SLEAP videos matches the list of files
        assert len(list_sleap_videos) == len(list_files)
