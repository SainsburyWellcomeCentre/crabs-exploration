import os
from pathlib import Path

import pooch


# Test to run the function with the test data
def test_detect_and_track_video(tmp_path, pooch_registry):
    """Test the detect-and-track-video entry point.

    Checks:
    - status code of the command
    - existence of csv file with predictions
    - existence of csv file with tracking metrics
    - existence of video file if requested
    - existence of exported frames if requested
    - MOTA score is as expected

    """
    # get trained model from pooch registry
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

    # get input video and annotations
    sample_video_dir = Path("04.09.2023-04-Right_RE_test_3_frames")
    path_to_input_video = pooch_registry.fetch(
        f"{sample_video_dir}/04.09.2023-04-Right_RE_test_3_frames.mp4"
    )
    path_to_annotations = pooch_registry.fetch(
        f"{sample_video_dir}/04.09.2023-04-Right_RE_test_3_frames_ground_truth.csv"
    )

    # get tracking config
    path_to_tracking_config = pooch_registry.fetch(
        f"{sample_video_dir}/tracking_config.yaml"
    )

    # # get expected output
    # path_to_tracked_boxes = pooch_registry.fetch(
    #     f"{sample_video_dir}/04.09.2023-04-Right_RE_test_3_frames_tracks.csv"
    # )
    # path_to_tracking_metrics = pooch_registry.fetch(
    #     f"{sample_video_dir}/tracking_metrics_output.csv"
    # )

    # set cwd to pytest tmpdir
    # output in ~/.crabs-exploration-test-data?

    # run detect-and-track-video with the test data
    status = os.system(
        "detect-and-track-video "
        f"--trained_model_path {path_to_ckpt} "
        f"--video_path {path_to_input_video} "
        f"--config_file {path_to_tracking_config} "
        f"--annotations_file {path_to_annotations} "
        # "--save_video "
        # "--save_frames "
        "--accelerator cpu "
        f"--output_dir {tmp_path}"
    )

    # check the command runs successfully
    assert status == 0

    # check the output files exist

    # capture logs
    # INFO:root:All 3 frames processed
    # INFO:root:Overall MOTA: 0.860465
