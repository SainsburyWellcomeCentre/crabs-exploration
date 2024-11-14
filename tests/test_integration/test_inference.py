import re
import subprocess
from pathlib import Path

import cv2
import pooch
import pytest

from crabs.tracker.utils.io import open_video


@pytest.fixture()
def input_data_paths(pooch_registry: pooch.Pooch):
    """Input data for a detector+tracking run.

    The data is fetched from the pooch registry.

    Returns
    -------
    dict
        Dictionary with the paths to the input video, annotations,
        tracking configuration and trained model.

    """
    input_data_paths = {}
    video_root_name = "04.09.2023-04-Right_RE_test_3_frames"
    input_data_paths["video_root_name"] = video_root_name

    # get trained model from pooch registry
    # download and unzip ml-runs
    list_files_ml_runs = pooch_registry.fetch(
        "ml-runs.zip",
        processor=pooch.Unzip(extract_dir=""),
        progressbar=True,
    )
    # get path to the last checkpoint
    input_data_paths["ckpt"] = next(
        x for x in list_files_ml_runs if x.endswith("last.ckpt")
    )

    # get input video, annotations and config from registry
    map_key_to_filepath = {
        "video": f"{video_root_name}/{video_root_name}.mp4",
        "annotations": f"{video_root_name}/{video_root_name}_ground_truth.csv",
        "tracking_config": f"{video_root_name}/tracking_config.yaml",
    }
    for key, filepath in map_key_to_filepath.items():
        input_data_paths[key] = pooch_registry.fetch(filepath)

    return input_data_paths


@pytest.mark.parametrize(
    "flags_to_append",
    [
        [],
        ["--save_video"],
        ["--save_frames"],
        ["--save_video", "--save_frames"],
    ],
)
def test_detect_and_track_video(
    input_data_paths: dict,
    tmp_path: Path,
    flags_to_append: list,
):
    """Test the detect-and-track-video entry point when groundtruth is passed.

    The test checks:
    - the exit code of the detect-and-track-video command
    - the existence of csv file with predictions
    - the existence of csv file with tracking metrics
    - the existence of video file if requested
    - the existence of exported frames if requested

    """
    # Define main detect-and-track-video command
    main_command = [
        "detect-and-track-video",
        f"--trained_model_path={input_data_paths['ckpt']}",
        f"--video_path={input_data_paths['video']}",
        f"--config_file={input_data_paths['tracking_config']}",
        f"--annotations_file={input_data_paths['annotations']}",
        "--accelerator=cpu",
    ]
    # append required flags
    main_command.extend(flags_to_append)

    # run command
    completed_process = subprocess.run(
        main_command,
        check=True,
        cwd=tmp_path,
        # set cwd to Pytest's temporary directory
        # so the output is saved there
    )

    # check the command runs successfully
    assert completed_process.returncode == 0

    # check the tracking output directory is created and has expected name
    output_dir_name_expected = "tracking_output"
    expected_pattern = re.compile(
        rf"{output_dir_name_expected}_\d{{8}}_\d{{6}}$"
    )
    list_cwd_subdirs = [x for x in tmp_path.iterdir() if x.is_dir()]
    tracking_output_dir = list_cwd_subdirs[0]
    assert len(list_cwd_subdirs) == 1
    assert expected_pattern.match(tracking_output_dir.stem)

    # check csv with predictions exists
    predictions_csv = (
        tmp_path
        / tracking_output_dir
        / f"{input_data_paths['video_root_name']}_tracks.csv"
    )
    assert (predictions_csv).exists()

    # check csv with tracking metrics exists
    tracking_metrics_csv = (
        tmp_path / tracking_output_dir / "tracking_metrics_output.csv"
    )
    assert (tracking_metrics_csv).exists()

    # if the video is requested: check it exists
    if "--save_video" in flags_to_append:
        assert (
            tmp_path
            / tracking_output_dir
            / f"{input_data_paths['video_root_name']}_tracks.mp4"
        ).exists()

    # if the frames are requested: check they exist
    if "--save_frames" in flags_to_append:
        input_video_object = open_video(input_data_paths["video"])
        total_n_frames = int(input_video_object.get(cv2.CAP_PROP_FRAME_COUNT))

        # check frames subdirectory exists
        frames_subdir = (
            tmp_path
            / tracking_output_dir
            / f"{input_data_paths['video_root_name']}_frames"
        )
        assert frames_subdir.exists()

        # check files are named as expected
        expected_pattern = re.compile(r"frame_\d{8}.png")
        list_files = [x for x in frames_subdir.iterdir() if x.is_file()]

        assert len(list_files) == total_n_frames
        assert all(expected_pattern.match(x.name) for x in list_files)
