"""Track crabs in a video using a trained detector."""

import argparse
import logging
import os
import sys
from pathlib import Path

import cv2
import lightning
import numpy as np
import torch
import torchvision.transforms.v2 as transforms
import yaml  # type: ignore

from crabs.detector.models import FasterRCNN
from crabs.detector.utils.detection import (
    log_mlflow_metadata_as_info,
    set_mlflow_run_name,
    setup_mlflow_logger,
)
from crabs.tracker.evaluate_tracker import TrackerEvaluate
from crabs.tracker.sort import Sort
from crabs.tracker.utils.io import (
    close_csv_file,
    prep_csv_writer,
    prep_video_writer,
    release_video,
    save_required_output,
)
from crabs.tracker.utils.tracking import prep_sort


class Tracking:
    """Interface for tracking crabs on a video using a trained detector.

    Parameters
    ----------
    args : argparse.Namespace)
        Command-line arguments containing configuration settings.

    Attributes
    ----------
    args : argparse.Namespace
        The command-line arguments provided.
    video_path : str
        The path to the input video.
    sort_tracker : Sort
        An instance of the sorting algorithm used for tracking.

    """

    def __init__(self, args: argparse.Namespace) -> None:
        """Initialise the tracking interface with the given arguments."""
        self.args = args
        self.config_file = args.config_file
        self.video_path = args.video_path
        self.trained_model_path = self.args.trained_model_path
        self.device = "cuda" if self.args.accelerator == "gpu" else "cpu"

        self.setup()
        self.prep_outputs()

        self.sort_tracker = Sort(
            max_age=self.config["max_age"],
            min_hits=self.config["min_hits"],
            iou_threshold=self.config["iou_threshold"],
        )

        # MLflow experiment name and run name
        self.experiment_name = args.experiment_name
        self.run_name = set_mlflow_run_name()
        self.mlflow_folder = args.mlflow_folder

        # Log MLflow information to screen
        log_mlflow_metadata_as_info(self)

    def setup_trainer(self):
        """Set up trainer object with logging for testing."""
        # Setup logger
        mlf_logger = setup_mlflow_logger(
            experiment_name=self.experiment_name,
            run_name=self.run_name,
            mlflow_folder=self.mlflow_folder,
            cli_args=self.args,
        )

        # Add trained model section to MLflow hyperparameters
        mlf_logger.log_hyperparams(
            {
                "trained_model/experiment_name": self.trained_model_expt_name,
                "trained_model/run_name": self.trained_model_run_name,
                "trained_model/ckpt_file": Path(self.trained_model_path).name,
            }
        )

        # Add other unlogged information from init?

        # Return trainer linked to logger
        return lightning.Trainer(
            accelerator=self.accelerator,  # lightning accelerators
            logger=mlf_logger,
        )

    def setup(self):
        """Load tracking config, trained model and input video path."""
        with open(self.config_file) as f:
            self.config = yaml.safe_load(f)

        # Get trained model
        self.trained_model = FasterRCNN.load_from_checkpoint(
            self.trained_model_path
        )
        self.trained_model.eval()
        self.trained_model.to(self.device)

        # Load the input video
        self.video = cv2.VideoCapture(self.video_path)
        if not self.video.isOpened():
            raise Exception("Error opening video file")
        self.video_file_root = f"{Path(self.video_path).stem}"

    def prep_outputs(self):
        """Prepare csv writer and if required, video writer."""
        (
            self.csv_writer,
            self.csv_file,
            self.tracking_output_dir,
        ) = prep_csv_writer(self.args.output_dir, self.video_file_root)

        if self.args.save_video:
            frame_width = int(self.video.get(cv2.CAP_PROP_FRAME_WIDTH))
            frame_height = int(self.video.get(cv2.CAP_PROP_FRAME_HEIGHT))
            cap_fps = self.video.get(cv2.CAP_PROP_FPS)

            self.video_output = prep_video_writer(
                self.tracking_output_dir,
                self.video_file_root,
                frame_width,
                frame_height,
                cap_fps,
            )
        else:
            self.video_output = None

    def get_prediction(self, frame: np.ndarray) -> torch.Tensor:
        """Get prediction from the trained model for a given frame.

        Parameters
        ----------
        frame : np.ndarray
            The input frame for which prediction is to be obtained.

        Returns
        -------
        torch.Tensor:
            The prediction tensor from the trained model.

        """
        transform = transforms.Compose(
            [
                transforms.ToImage(),
                transforms.ToDtype(torch.float32, scale=True),
            ]
        )
        img = transform(frame).to(self.device)
        img = img.unsqueeze(0)
        with torch.no_grad():
            prediction = self.trained_model(img)
        return prediction

    def update_tracking(self, prediction: dict) -> np.ndarray:
        """Update the tracking system with the latest prediction.

        Parameters
        ----------
        prediction : dict
            Dictionary containing predicted bounding boxes, scores, and labels.

        Returns
        -------
        np.ndarray:
            tracked bounding boxes after updating the tracking system.

        """
        pred_sort = prep_sort(prediction, self.config["score_threshold"])
        tracked_boxes_id_per_frame = self.sort_tracker.update(pred_sort)
        self.tracked_bbox_id.append(tracked_boxes_id_per_frame)

        return tracked_boxes_id_per_frame

    def run_tracking(self):
        """Run object detection + tracking on the video frames."""
        # If we pass ground truth: check the path exist
        if self.args.annotations_file and not os.path.exists(
            self.args.annotations_file
        ):
            logging.info(
                f"Ground truth file {self.args.annotations_file} "
                "does not exist."
                "Exiting..."
            )
            return

        # initialisation
        frame_idx = 0
        self.tracked_bbox_id = []

        # Loop through frames of the video in batches
        while self.video.isOpened():
            # Break if beyond end frame (mostly for debugging)
            if (
                self.args.max_frames_to_read
                and frame_idx + 1 > self.args.max_frames_to_read
            ):
                break

            # get total n frames
            total_frames = int(self.video.get(cv2.CAP_PROP_FRAME_COUNT))

            # read frame
            ret, frame = self.video.read()
            if not ret and (frame_idx == total_frames):
                logging.info(f"All {total_frames} frames processed")
                break
            elif not ret:
                logging.info(
                    f"Cannot read frame {frame_idx+1}/{total_frames}. "
                    "Exiting..."
                )
                break

            # predict bounding boxes
            prediction = self.get_prediction(frame)
            pred_scores = prediction[0]["scores"].detach().cpu().numpy()

            # run tracking
            tracked_boxes_id_per_frame = self.update_tracking(prediction)
            save_required_output(
                self.video_file_root,
                self.args.save_frames,
                self.tracking_output_dir,
                self.csv_writer,
                self.args.save_video,
                self.video_output,
                tracked_boxes_id_per_frame,
                frame,
                frame_idx + 1,
                pred_scores,
            )

            # update frame number
            frame_idx += 1

        if self.args.annotations_file:
            evaluation = TrackerEvaluate(
                self.args.annotations_file,
                self.tracked_bbox_id,
                self.config["iou_threshold"],
            )
            evaluation.run_evaluation()

        # Close input video
        self.video.release()

        # Close outputs
        if self.args.save_video:
            release_video(self.video_output)

        if self.args.save_frames:
            close_csv_file(self.csv_file)


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
    inference.run_tracking()


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
        "--output_dir",
        type=str,
        default="tracking_output",
        help=(
            "Root name of the directory to save the tracking output. "
            "The name of the output directory is appended with a timestamp. "
            "Default: ./tracking_output_<timestamp>. "
        ),
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
        "--accelerator",
        type=str,
        default="gpu",
        help=(
            "Accelerator for Pytorch. "
            "Valid inputs are: cpu or gpu. Default: gpu."
        ),
    )
    parser.add_argument(
        "--experiment_name",
        type=str,
        default="Inference",
        help=(
            "Name of the experiment in MLflow, under which the current run "
            "will be logged. "
            "By default: Inference."
        ),
    )
    parser.add_argument(
        "--mlflow_folder",
        type=str,
        default="./ml-runs",
        help=(
            "Path to MLflow directory where to log the evaluation data. "
            "Default: 'ml-runs' directory under the current working directory."
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
