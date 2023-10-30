from pathlib import Path

import pytest
import sys
import subprocess


from bboxes_labelling.extract_frames_to_label_w_sleap import get_list_of_sleap_videos


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


# --------------------------
# TODO: make fixtures
script_path = (
    Path(__file__).parent
    / ".."
    / "bboxes labelling"
    / "extract_frames_to_label_w_sleap.py"
)

python_interpreter = sys.executable

# TODO: add a very small sample clip!
sample_input_video = (
    Path(__file__).parent / "data" / "NINJAV_S001_S001_T003_subclip.mp4"
)
sample_input_dir = Path(__file__).parent / "data"
# TODO: use a temp directory from pytest fixtures?
sample_output_dir = Path(__file__).parent / "output"

# TODO: have this as a fixture? or parametrise?
sample_input_params = {
    "output_path": sample_output_dir,
    "video_extensions": "mp4",
    "initial_samples": "5",
    "scale": "0.5",
    "n_components": "3",
    "n_clusters": "5",
    "per_cluster": "1",
}
# ----------


# TODO: add to class if required
def test_help():
    result = subprocess.run(
        [python_interpreter, script_path, "-h"],
        capture_output=True,
        # text=True
    )
    # result.args, result.returncode, result.stdout, result.stderr
    assert result.returncode == 0


# TODO: add to class if required
def test_small_frame_extraction_one_video():
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
            python_interpreter,
            script_path,
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


def test_small_frame_extraction_one_dir():
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
            python_interpreter,
            script_path,
            sample_input_dir,
        ]
        + list_cli_args,
        capture_output=True,
        text=True,
    )

    # check return code
    assert result.returncode == 0

    # check name of files
