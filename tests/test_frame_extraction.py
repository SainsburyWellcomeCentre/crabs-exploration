from pathlib import Path
import sys
import subprocess


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


def test_help():
    result = subprocess.run(
        [python_interpreter, script_path, "-h"],
        capture_output=True,
        # text=True
    )
    # result.args, result.returncode, result.stdout, result.stderr
    assert result.returncode == 0


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
