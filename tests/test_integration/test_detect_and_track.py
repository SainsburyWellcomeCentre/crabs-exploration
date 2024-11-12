import re
import subprocess
from pathlib import Path

import cv2
import pooch
import pytest

from crabs.tracker.utils.io import open_video


@pytest.mark.parametrize(
    "flags_to_append",
    [
        [],
        ["--save_video"],
        ["--save_frames"],
        ["--save_video --save_frames"],
    ],
)
def test_detect_and_track_video(
    pooch_registry: pooch.Pooch, tmp_path: Path, flags_to_append: list
):
    """Test the detect-and-track-video entry point.

    Checks:
    - status code of the command
    - existence of csv file with predictions
    - existence of csv file with tracking metrics
    - existence of video file if requested
    - existence of exported frames if requested
    - MOTA score is as expected

    """
    # get trained model from pooch registry -----> make fixture
    list_files_ml_runs = pooch_registry.fetch(
        "ml-runs.zip",
        processor=pooch.Unzip(
            extract_dir="",
        ),
        progressbar=True,
    )
    path_to_ckpt = [
        file for file in list_files_ml_runs if file.endswith("last.ckpt")
    ][0]

    # get input video, annotations and config ----> make fixture
    video_root_name = "04.09.2023-04-Right_RE_test_3_frames"
    path_to_input_video = pooch_registry.fetch(
        f"{video_root_name}/{video_root_name}.mp4"
    )
    path_to_annotations = pooch_registry.fetch(
        f"{video_root_name}/{video_root_name}_ground_truth.csv"
    )
    path_to_tracking_config = pooch_registry.fetch(
        f"{video_root_name}/tracking_config.yaml"
    )

    # Open input video
    input_video_object = open_video(path_to_input_video)
    total_n_frames = int(input_video_object.get(cv2.CAP_PROP_FRAME_COUNT))

    # # get expected output
    # path_to_tracked_boxes = pooch_registry.fetch(
    #     f"{sample_video_dir}/04.09.2023-04-Right_RE_test_3_frames_tracks.csv"
    # )
    # path_to_tracking_metrics = pooch_registry.fetch(
    #     f"{sample_video_dir}/tracking_metrics_output.csv"
    # )

    # run detect-and-track-video with the test data
    main_command = [
        "detect-and-track-video",
        f"--trained_model_path={path_to_ckpt}",
        f"--video_path={path_to_input_video}",
        f"--config_file={path_to_tracking_config}",
        f"--annotations_file={path_to_annotations}",
        "--accelerator=cpu",
        # f"--output_dir={tmp_path}",
    ]
    main_command.extend(flags_to_append)
    completed_process = subprocess.run(
        main_command,
        check=True,
        cwd=tmp_path,  # set cwd to pytest tmpdir if no output_dir is passed
    )

    # check the command runs successfully
    assert completed_process.returncode == 0

    # check the tracking output directory is created
    pattern = re.compile(r"tracking_output_\d{8}_\d{6}")
    list_subdirs = [x for x in tmp_path.iterdir() if x.is_dir()]
    tracking_output_dir = list_subdirs[0]
    assert len(list_subdirs) == 1
    assert pattern.match(tracking_output_dir.stem)

    # check csv with predictions exists
    assert (
        tmp_path / tracking_output_dir / f"{video_root_name}_tracks.csv"
    ).exists()

    # check csv with tracking metrics exists
    assert (
        tmp_path / tracking_output_dir / "tracking_metrics_output.csv"
    ).exists()

    # if the video is requested: check it exists
    if "--save_video" in flags_to_append:
        assert (
            tmp_path / tracking_output_dir / f"{video_root_name}_tracks.mp4"
        ).exists()

    # if the frames are requested: check they exist
    if "--save_frames" in flags_to_append:
        # check subdirectory exists
        frames_subdir = (
            tmp_path / tracking_output_dir / f"{video_root_name}_frames"
        )
        assert frames_subdir.exists()

        # check files
        pattern = re.compile(r"frame_\d{8}.png")
        list_files = [x for x in frames_subdir.iterdir() if x.is_file()]

        assert len(list_files) == total_n_frames
        assert all(pattern.match(x.name) for x in list_files)

    # check the MOTA score is as expected
    # capture logs
    # INFO:root:All 3 frames processed
    # INFO:root:Overall MOTA: 0.860465
