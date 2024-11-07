"""Track crabs in a video using a trained detector."""

import argparse
import logging
import sys
from datetime import datetime
from pathlib import Path

import cv2
import numpy as np
import torch
import torchvision.transforms.v2 as transforms
import yaml  # type: ignore

from crabs.detector.models import FasterRCNN
from crabs.detector.utils.evaluate import (
    get_config_from_ckpt,
    get_mlflow_parameters_from_ckpt,
)
from crabs.tracker.evaluate_tracker import TrackerEvaluate
from crabs.tracker.sort import Sort
from crabs.tracker.utils.io import (
    get_video_parameters,
    write_frame_as_image,
    write_frame_to_output_video,
    write_tracked_detections_to_csv,
)
from crabs.tracker.utils.tracking import format_bbox_predictions_for_sort


class Tracking:
    """Interface for tracking crabs on a video using a trained detector.

    Parameters
    ----------
    args : argparse.Namespace)
        Command-line arguments containing configuration settings.

    """

    def __init__(self, args: argparse.Namespace) -> None:
        """Initialise the tracking interface with the given arguments."""
        # inputs
        self.args = args
        self.config_file = args.config_file
        self.load_config_yaml()

        # trained model data
        self.trained_model_path = args.trained_model_path
        trained_model_params = get_mlflow_parameters_from_ckpt(
            self.trained_model_path
        )
        self.trained_model_run_name = trained_model_params["run_name"]
        self.trained_model_expt_name = trained_model_params[
            "cli_args/experiment_name"
        ]
        self.trained_model_config = get_config_from_ckpt(
            config_file=None,
            trained_model_path=self.trained_model_path,
        )

        # input video data
        self.input_video_path = args.video_path
        self.input_video_file_root = f"{Path(self.input_video_path).stem}"
        self.input_video_object = cv2.VideoCapture(self.input_video_path)
        if not self.input_video_object.isOpened():
            raise Exception("Error opening video file")
        self.input_video_params = get_video_parameters(self.input_video_object)

        # output directory root name
        self.tracking_output_dir_root = args.output_dir
        self.frame_name_format_str = "frame_{frame_idx:08d}.png"

        # hardware
        self.accelerator = "cuda" if args.accelerator == "gpu" else "cpu"

        # Prepare outputs
        # - output directory
        # - csv file
        # - video writer if required
        # - frames subdirectory if required
        self.prep_outputs()

    def load_config_yaml(self):
        """Load yaml file that contains config parameters."""
        with open(self.config_file) as f:
            self.config = yaml.safe_load(f)

    def prep_outputs(self):
        """Prepare output directories and files."""
        # Create output directory
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.tracking_output_dir = Path(
            self.tracking_output_dir_root + f"_{timestamp}"
        )
        self.tracking_output_dir.mkdir(parents=True, exist_ok=True)

        # Set name of output csv file
        self.csv_file_path = str(
            self.tracking_output_dir
            / f"{self.input_video_file_root}_tracks.csv"
        )

        # Set up output video writer if required
        if self.args.save_video:
            output_codec = cv2.VideoWriter_fourcc("m", "p", "4", "v")
            self.output_video_path = str(
                self.tracking_output_dir
                / f"{self.input_video_file_root}_tracks.mp4"
            )
            self.output_video_object = cv2.VideoWriter(
                self.output_video_path,
                output_codec,
                self.input_video_params["fps"],
                (
                    self.input_video_params["frame_width"],
                    self.input_video_params["frame_height"],
                ),
            )

        # Set up frames subdirectory if required
        if self.args.save_frames:
            self.frames_subdir = (
                self.tracking_output_dir
                / f"{self.input_video_file_root}_frames"
            )
            self.frames_subdir.mkdir(parents=True, exist_ok=True)

    def parse_frame_reading_error_and_log(self, frame_idx, total_frames):
        """Parse error message for reading frames."""
        if frame_idx == total_frames:
            logging.info(f"All {total_frames} frames processed")
        else:
            logging.info(
                f"Error reading frame index " f"{frame_idx}/{total_frames}."
            )

    def update_tracking(self, prediction: dict) -> np.ndarray:
        """Update the tracking data with the latest prediction.

        Parameters
        ----------
        prediction : dict
            Dictionary containing predicted bounding boxes, scores, and labels.
            # What are the keys?

        Returns
        -------
        np.ndarray:
            tracked bounding boxes after updating the tracking system.

        """
        # format predictions for SORT
        pred_sort = format_bbox_predictions_for_sort(
            prediction, self.config["score_threshold"]
        )

        # update tracked bboxes and append
        tracked_boxes_id_per_frame = self.sort_tracker.update(pred_sort)

        return tracked_boxes_id_per_frame

    def core_detection_and_tracking(self, transform, trained_model):
        """Run detection and tracking."""
        # Run detection+tracking per frame
        # TODO: factor out?
        frame_idx = 0
        list_tracked_bboxes_all_frames = []
        list_tracked_bboxes_scores = []
        while self.input_video_object.isOpened():
            # Read frame
            ret, frame = self.input_video_object.read()
            if not ret:
                self.parse_frame_reading_error_and_log(
                    frame_idx, self.input_video_params["total_frames"]
                )
                break

            # Run prediction per frame
            # TODO: can I pass a video as a generator?
            # do I need to go frame by frame?
            # TODO: use trainer.predict()
            image_tensors = transform(frame).to(self.accelerator)
            image_tensors = image_tensors.unsqueeze(0)
            with torch.no_grad():
                prediction = trained_model(image_tensors)
                list_tracked_bboxes_scores.append(
                    prediction[0]["scores"].detach().cpu().numpy()
                )

            # Update tracking
            tracked_boxes_id_per_frame = self.update_tracking(prediction)
            list_tracked_bboxes_all_frames.append(tracked_boxes_id_per_frame)

            # Write frame with tracks to video if required
            if self.args.save_video:
                write_frame_to_output_video(
                    frame,
                    tracked_boxes_id_per_frame,
                    self.output_video_object,  # ---- set up
                )

            # Save frame without detections if required
            if self.args.save_frames:
                frame_path = str(
                    self.frames_subdir
                    / self.frame_name_format_str.format(frame_idx=frame_idx)
                )
                write_frame_as_image(frame, frame_path)

            # Update frame index
            frame_idx += 1

        # Close videos
        self.input_video_object.release()
        if self.args.save_video:
            self.output_video_object.release()

        # TODO: instead return a dict, with key=frame_index?
        return list_tracked_bboxes_all_frames, list_tracked_bboxes_scores

    def detect_and_track_video(self) -> None:
        """Run detection and tracking on input video."""
        # Load trained model
        trained_model = FasterRCNN.load_from_checkpoint(
            self.trained_model_path,
            config=self.trained_model_config,  # config of trained model!
        )
        trained_model.eval()
        trained_model.to(self.accelerator)

        # Initialise SORT tracker
        self.sort_tracker = Sort(
            max_age=self.config["max_age"],
            min_hits=self.config["min_hits"],
            iou_threshold=self.config["iou_threshold"],
        )

        # Define transforms
        transform = transforms.Compose(
            [
                transforms.ToImage(),
                transforms.ToDtype(torch.float32, scale=True),
            ]
        )

        # Run detection and tracking
        # TODO: make output a dict
        (list_tracked_bboxes_all_frames, list_tracked_bboxes_scores) = (
            self.core_detection_and_tracking(transform, trained_model)
        )

        # Write list of tracked bounding boxes to csv
        write_tracked_detections_to_csv(
            self.csv_file_path,
            list_tracked_bboxes_all_frames,
            list_tracked_bboxes_scores,
        )

        # Write to video if required -- outside loop

        # Write frames if required -- outside loop

        # Evaluate tracker if ground truth is passed
        # TODO: refactor?
        if self.args.annotations_file:
            evaluation = TrackerEvaluate(
                self.args.annotations_file,
                list_tracked_bboxes_all_frames,
                self.config["iou_threshold"],
                self.tracking_output_dir,
            )
            evaluation.run_evaluation()


def main(args) -> None:
    """Run detection+tracking inference on video.

    Parameters
    ----------
    args : argparse
        Arguments or configuration settings for testing.

    Returns
    -------
        None

    """
    inference = Tracking(args)

    inference.detect_and_track_video()


def tracking_parse_args(args):
    """Parse command-line arguments for tracking."""
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--trained_model_path",
        type=str,
        required=True,
        help="Location of trained model (a .ckpt file). ",
    )
    parser.add_argument(
        "--video_path",
        type=str,
        required=True,
        help="Location of the video to be tracked.",
    )
    parser.add_argument(
        "--config_file",
        type=str,
        default=str(Path(__file__).parent / "config" / "tracking_config.yaml"),
        help=(
            "Location of YAML config to control tracking. "
            "Default: "
            "crabs-exploration/crabs/tracking/config/tracking_config.yaml. "
        ),
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="tracking_output",
        help=(
            "Root name of the directory to save the tracking output. "
            "The name of the output directory is appended with a timestamp. "
            "The tracking output consist of a .csv. file named "
            "<video-name>_tracks.csv with the tracked bounding boxes. "
            "Optionally, it can include a video file named "
            "<video-name>_tracks.mp4, and all frames from the video "
            "under a <video-name>_frames subdirectory. "
            "Default: ./tracking_output_<timestamp>. "
        ),
    )
    parser.add_argument(
        "--save_video",
        action="store_true",
        help=(
            "Add a video with tracked bounding boxes "
            "to the tracking output directory. "
            "The tracked video is called <input-video-name>_tracks.mp4. "
        ),
    )
    parser.add_argument(
        "--save_frames",
        action="store_true",
        help=(
            "Add all frames to the tracking output. "
            "The frames are saved as-is, without bounding boxes, to "
            "support their visualisation and correction using the VIA tool. "
        ),
    )
    parser.add_argument(
        "--annotations_file",
        type=str,
        default=None,
        help=(
            "Location of JSON file containing ground truth annotations "
            "(optional). "
            "If passed, the evaluation metrics for the tracker are computed."
        ),
    )
    parser.add_argument(
        "--accelerator",
        type=str,
        default="gpu",
        help=(
            "Accelerator for Pytorch. "
            "Valid inputs are: cpu or gpu. Default: gpu."
        ),
    )
    parser.add_argument(
        "--max_frames_to_read",
        type=int,
        default=None,
        help=(
            "Debugging option to limit "
            "the maximum number of frames to read in the video. "
            "It affects all the tracking outputs (csv, frames and video) "
            "and the MOTA computation, which will be restricted to just "
            "the first N frames. "
        ),
    )
    return parser.parse_args(args)


def app_wrapper():
    """Wrap function to run the tracking application."""
    logging.getLogger().setLevel(logging.INFO)

    torch.set_float32_matmul_precision("medium")

    tracking_args = tracking_parse_args(sys.argv[1:])
    main(tracking_args)


if __name__ == "__main__":
    app_wrapper()
