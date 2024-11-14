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
    generate_tracked_video,
    open_video,
    parse_video_frame_reading_error_and_log,
    write_all_video_frames_as_images,
    write_tracked_detections_to_csv,
)
from crabs.tracker.utils.tracking import (
    format_and_filter_bbox_predictions_for_sort,
)

DEFAULT_TRACKING_CONFIG = str(
    Path(__file__).parent / "config" / "tracking_config.yaml"
)


class Tracking:
    """Interface for detecting and tracking crabs on a video.

    Parameters
    ----------
    args : argparse.Namespace
        Command-line arguments containing configuration settings.

    """

    def __init__(self, args: argparse.Namespace) -> None:
        """Initialise the tracking interface with the given arguments."""
        # CLI inputs and config file
        self.args = args
        self.config_file = args.config_file
        self.load_config_yaml()

        # trained model data
        self.trained_model_path = args.trained_model_path
        trained_model_params = get_mlflow_parameters_from_ckpt(
            self.trained_model_path
        )
        # to log later in MLflow:
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

        # tracking output directory root name
        self.tracking_output_dir_root = args.output_dir
        self.frame_name_format_str = "frame_{frame_idx:08d}.png"

        # hardware
        self.accelerator = "cuda" if args.accelerator == "gpu" else "cpu"

        # Prepare outputs:
        # output directory, csv, and if required video and frames
        self.prep_outputs()

    def load_config_yaml(self):
        """Load yaml file that contains config parameters."""
        with open(self.config_file) as f:
            self.config = yaml.safe_load(f)

    def prep_outputs(self):
        """Prepare output directory and file paths.

        This method:
        - creates a timestamped directory to store the tracking output.
        - sets the name of the output csv file for the tracked bounding boxes.
        - sets up the output video path if required.
        - sets up the frames subdirectory path if required.
        """
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

        # Set up output video path if required
        if self.args.save_video:
            self.output_video_path = str(
                self.tracking_output_dir
                / f"{self.input_video_file_root}_tracks.mp4"
            )

        # Set up frames subdirectory path if required
        if self.args.save_frames:
            self.frames_subdir = (
                self.tracking_output_dir
                / f"{self.input_video_file_root}_frames"
            )
            self.frames_subdir.mkdir(parents=True, exist_ok=True)

    def prep_detector_and_tracker(self):
        """Prepare the trained detector and the tracker for inference."""
        # TODO: use Lightning's Trainer?

        # Load trained model
        self.trained_model = FasterRCNN.load_from_checkpoint(
            self.trained_model_path,
            config=self.trained_model_config,  # config of trained model!
        )
        self.trained_model.eval()
        self.trained_model.to(self.accelerator)

        # Define transforms to apply to input frames
        self.inference_transforms = transforms.Compose(
            [
                transforms.ToImage(),
                transforms.ToDtype(torch.float32, scale=True),
            ]
        )

        # Initialise SORT tracker
        self.sort_tracker = Sort(
            max_age=self.config["max_age"],
            min_hits=self.config["min_hits"],
            iou_threshold=self.config["iou_threshold"],
        )

    def run_tracking(self, prediction_dict: dict) -> np.ndarray:
        """Update the tracker with the latest prediction.

        Parameters
        ----------
        prediction_dict : dict
            Dictionary with data of the predicted bounding boxes.
            The keys are: "boxes", "scores", and "labels". The labels
            refer to the class of the object detected, and not its ID.

        Returns
        -------
        np.ndarray:
            Array of tracked bounding boxes with object IDs added as the last
            column. The shape of the array is (n, 5), where n is the number of
            tracked boxes. The columns correspond to the values (xmin, ymin,
            xmax, ymax, id).

        """
        # format predictions for SORT
        prediction_tensor = format_and_filter_bbox_predictions_for_sort(
            prediction_dict, self.config["score_threshold"]
        )

        # update tracked bboxes and append
        tracked_boxes_id_per_frame = self.sort_tracker.update(
            prediction_tensor.cpu()  # move to CPU for SORT
        )

        return tracked_boxes_id_per_frame

    def run_detection(self, frame: np.ndarray) -> dict:
        """Run detection on a single frame.

        Returns
        -------
        dict:
            Dictionary with data of the predicted bounding boxes.
            The keys are "boxes", "scores", and "labels". The labels
            refer to the class of the object detected, and not its ID.
            The data is stored as torch tensors.

        """
        # Apply transforms to frame and place tensor on devide
        image_tensor = self.inference_transforms(frame).to(self.accelerator)

        # Add batch dimension
        image_tensor = image_tensor.unsqueeze(0)

        # Run detection
        with torch.no_grad():
            # use [0] to select the one image in the batch
            detections_dict = self.trained_model(image_tensor)[0]

        return detections_dict

    def core_detection_and_tracking(self):
        """Run detection and tracking loop through all video frames.

        Returns a dictionary with tracked bounding boxes per frame, and
        with scores for each detection.

        Returns
        -------
        dict:
            A nested dictionary that maps frame indices (0-based) to a
            dictionary with the following keys:
            - "tracked_boxes", which contains the tracked bounding boxes as a
            numpy array of shape (n, 5), where n is the number of tracked
            boxes, and the 5 columns correspond to the values (xmin, ymin,
            xmax, ymax, id).
            - "scores", which contains the scores for each bounding box,
            as a numpu array of shape (nboxes,)

        """
        # Initialise dict to store tracked bboxes
        tracked_detections_all_frames = {}

        # Open input video
        input_video_object = open_video(self.input_video_path)
        total_n_frames = int(input_video_object.get(cv2.CAP_PROP_FRAME_COUNT))

        # Loop over frames
        frame_idx = 0
        while input_video_object.isOpened():
            # Read frame
            ret, frame = input_video_object.read()
            if not ret:
                parse_video_frame_reading_error_and_log(
                    frame_idx, total_n_frames
                )
                break

            # Run detection per frame
            detections_dict = self.run_detection(frame)

            # Update tracking
            tracked_boxes_array = self.run_tracking(detections_dict)

            # Add data to dict; key is frame index (0-based) for input clip
            tracked_detections_all_frames[frame_idx] = {
                "tracked_boxes": tracked_boxes_array[:, :-1],
                "ids": tracked_boxes_array[:, -1],  # IDs are the last column
                "scores": detections_dict["scores"],
            }

            # Update frame index
            frame_idx += 1

        # Release video object
        input_video_object.release()

        return tracked_detections_all_frames

    def detect_and_track_video(self) -> None:
        """Run detection and tracking on input video."""
        # Prepare detector and tracker
        # - Load trained model
        # - Define transforms
        # - Initialise SORT tracker
        self.prep_detector_and_tracker()

        # Run detection and tracking over all frames in video
        tracked_bboxes_dict = self.core_detection_and_tracking()

        # Write list of tracked bounding boxes to csv
        write_tracked_detections_to_csv(
            self.csv_file_path,
            tracked_bboxes_dict,
            frame_name_regexp=self.frame_name_format_str,
        )

        # Generate tracked video if required
        # (it loops again thru frames)
        if self.args.save_video:
            generate_tracked_video(
                self.input_video_path,
                self.output_video_path,
                tracked_bboxes_dict,
            )
            logging.info(f"Tracked video saved to {self.output_video_path}")

        # Write frames if required
        # (it loops again thru frames)
        if self.args.save_frames:
            write_all_video_frames_as_images(
                self.input_video_path,
                self.frames_subdir,
                self.frame_name_format_str,
            )
            logging.info(
                "Input frames saved to "
                f"{self.tracking_output_dir / self.frames_subdir}"
            )

        # Evaluate tracker if ground truth is passed
        if self.args.annotations_file:
            evaluation = TrackerEvaluate(
                self.args.annotations_file,
                tracked_bboxes_dict,
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
        default=DEFAULT_TRACKING_CONFIG,
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
