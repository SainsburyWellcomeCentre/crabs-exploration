import argparse
import logging
import os
import sys
from pathlib import Path

import cv2
import numpy as np
import torch
import torchvision.transforms.v2 as transforms
import yaml  # type: ignore

from crabs.detector.models import FasterRCNN
from crabs.tracker.evaluate_tracker import TrackerEvaluate
from crabs.tracker.sort import Sort
from crabs.tracker.utils.io import (
    close_csv_file,
    prep_csv_writer,
    prep_video_writer,
    release_video,
    save_required_output,
)
from crabs.tracker.utils.tracking import (
    calculate_velocity,
    get_orientation,
    prep_sort,
)

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class Tracking:
    """
    A class for performing crabs tracking on a video
    using a trained model.

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
        self.args = args

        self.config_file = args.config_file
        self.load_config_yaml()  # TODO: load config from trained model (like in evaluation)?

        self.video_path = args.video_path
        self.video_file_root = f"{Path(self.video_path).stem}"
        self.trained_model_path = self.args.trained_model_path

        self.trained_model = self.load_trained_model()

        self.sort_tracker = Sort(
            max_age=self.config["max_age"],
            min_hits=self.config["min_hits"],
            iou_threshold=self.config["iou_threshold"],
        )

        (
            self.csv_writer,
            self.csv_file,
            self.tracking_output_dir,
        ) = prep_csv_writer(self.args.output_dir, self.video_file_root)

    def load_config_yaml(self):
        """
        Load yaml file that contains config parameters.
        """
        with open(self.config_file, "r") as f:
            self.config = yaml.safe_load(f)

    def load_trained_model(self) -> torch.nn.Module:
        """
        Load the trained model.

        Returns
        -------
        torch.nn.Module
        """
        # Get trained model
        trained_model = FasterRCNN.load_from_checkpoint(
            self.trained_model_path
        )
        trained_model.eval()
        trained_model.to(DEVICE)  # Should device be a CLI?
        return trained_model

    def load_video(self) -> None:
        """
        Load the input video, and prepare the output video if required.
        """
        self.video = cv2.VideoCapture(self.video_path)
        if not self.video.isOpened():
            raise Exception("Error opening video file")

        cap_fps = self.video.get(cv2.CAP_PROP_FPS)
        self.frame_time_interval = 1 / cap_fps

        if self.args.save_video:
            frame_width = int(self.video.get(cv2.CAP_PROP_FRAME_WIDTH))
            frame_height = int(self.video.get(cv2.CAP_PROP_FRAME_HEIGHT))

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
        """
        Get prediction from the trained model for a given frame.

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
        img = transform(frame).to(DEVICE)
        img = img.unsqueeze(0)
        with torch.no_grad():
            prediction = self.trained_model(img)
        return prediction

    def update_tracking(self, prediction: dict) -> list[list[float]]:
        """
        Update the tracking system with the latest prediction.

        Parameters
        ----------
        prediction : dict
            Dictionary containing predicted bounding boxes, scores, and labels.

        Returns
        -------
        list[list[float]]:
            list of tracked bounding boxes after updating the tracking system.
        """
        pred_sort = prep_sort(prediction, self.config["score_threshold"])
        tracked_boxes = self.sort_tracker.update(pred_sort)
        self.tracked_list.append(tracked_boxes)
        return tracked_boxes

    def run_tracking(self):
        """
        Run object detection + tracking on the video frames.
        """
        # If we pass ground truth: check the path exist
        if self.args.gt_path and not os.path.exists(self.args.gt_path):
            logging.info(
                f"Ground truth file {self.args.gt_path} does not exist. Exiting..."
            )
            return

        # In any case run inference
        # initialisation
        frame_number = 1
        self.tracked_list = []
        previous_positions = {}

        # Loop through frames of the video in batches
        while self.video.isOpened():
            # Break if beyond end frame (mostly for debugging)
            if (
                self.args.max_frames_to_read
                and frame_number > self.args.max_frames_to_read
            ):
                break

            # read frame
            ret, frame = self.video.read()
            if not ret:
                print("No frame read. Exiting...")
                break

            # predict bounding boxes
            prediction = self.get_prediction(frame)

            # run tracking
            tracked_boxes = self.update_tracking(prediction)

            velocities = calculate_velocity(
                tracked_boxes, previous_positions, self.frame_time_interval
            )

            orientation_data = get_orientation(tracked_boxes, velocities)
            # print(orientation_data)

            # # Display frame with arrows
            # cv2.imshow("Orientation Visualization", frame_with_arrows)
            # cv2.imwrite(f"frame_{frame_number}.jpg", frame_with_arrows)

            # if cv2.waitKey(1) & 0xFF == ord("q"):
            #     break

            save_required_output(
                self.video_file_root,
                self.args.save_frames,
                self.tracking_output_dir,
                self.csv_writer,
                self.args.save_video,
                self.video_output,
                tracked_boxes,
                frame,
                frame_number,
                orientation_data,
            )

            # update frame number
            frame_number += 1

        if self.args.gt_path:
            evaluation = TrackerEvaluate(
                self.args.gt_path,
                self.tracked_list,
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
    """
    Main function to run the inference on video based on the trained model.

    Parameters
    ----------
    args : argparse
        Arguments or configuration settings for testing.

    Returns
    -------
        None
    """

    inference = Tracking(args)
    inference.load_video()
    inference.run_tracking()


def tracking_parse_args(args):
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--trained_model_path",
        type=str,
        required=True,
        help="location of checkpoint of the trained model",
    )
    parser.add_argument(
        "--video_path",
        type=str,
        required=True,
        help="location of video to be tracked",
    )
    parser.add_argument(
        "--config_file",
        type=str,
        default=str(Path(__file__).parent / "config" / "tracking_config.yaml"),
        help=(
            "Location of YAML config to control tracking. "
            "Default: crabs-exploration/crabs/tracking/config/tracking_config.yaml"
        ),
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="crabs_track_output",
        help="Directory to save the track output",  # is this a csv or a video? (or both)
    )
    parser.add_argument(
        "--max_frames_to_read",
        type=int,
        default=None,
        help="Maximum number of frames to read (mostly for debugging).",
    )
    parser.add_argument(
        "--gt_path",
        type=str,
        default=None,
        help=(
            "Location of json file containing ground truth annotations (optional)."
            "If passed, evaluation metrics are computed."
        ),
    )
    parser.add_argument(
        "--save_video",
        action="store_true",
        help="Save video inference with tracking output",
    )
    parser.add_argument(
        "--save_frames",
        action="store_true",
        help="Save frame to be used in correcting track labelling",
    )
    return parser.parse_args(args)


def app_wrapper():
    torch.set_float32_matmul_precision("medium")

    tracking_args = tracking_parse_args(sys.argv[1:])
    main(tracking_args)


if __name__ == "__main__":
    app_wrapper()
