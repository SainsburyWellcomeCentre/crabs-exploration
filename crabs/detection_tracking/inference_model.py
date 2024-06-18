import argparse
import csv
import os
import sys
from pathlib import Path
from typing import Any, Optional, TextIO, Tuple

import cv2
import numpy as np
import torch
import torchvision.transforms.v2 as transforms
import yaml  # type: ignore

from crabs.detection_tracking.models import FasterRCNN
from crabs.detection_tracking.sort import Sort
from crabs.detection_tracking.tracking_utils import (
    evaluate_mota,
    get_ground_truth_data,
    save_frame_and_csv,
    write_tracked_bbox_to_csv,
)
from crabs.detection_tracking.visualization import draw_bbox


class DetectorInference:
    """
    A class for performing object detection or tracking inference on a video
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
        self.video_path = args.video_path

        self.video_file_root = f"{Path(self.video_path).stem}"
        self.trained_model = self.load_trained_model()
        self.load_config_yaml()
        self.sort_tracker = Sort(
            max_age=self.config["max_age"],
            min_hits=self.config["min_hits"],
            iou_threshold=self.config["iou_threshold"],
        )

    def load_config_yaml(self):
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
            self.args.checkpoint_path
        )
        trained_model.eval()
        trained_model.to(self.args.accelerator)
        return trained_model

    def prep_sort(self, prediction: dict) -> np.ndarray:
        """
        Put predictions in format expected by SORT

        Parameters
        ----------
        prediction : dict
            The dictionary containing predicted bounding boxes, scores, and labels.

        Returns
        -------
        np.ndarray:
            An array containing sorted bounding boxes of detected objects.
        """
        pred_boxes = prediction[0]["boxes"].detach().cpu().numpy()
        pred_scores = prediction[0]["scores"].detach().cpu().numpy()
        pred_labels = prediction[0]["labels"].detach().cpu().numpy()

        pred_sort = []
        for box, score, label in zip(pred_boxes, pred_scores, pred_labels):
            if score > self.config["score_threshold"]:
                bbox = np.concatenate((box, [score]))
                pred_sort.append(bbox)

        return np.asarray(pred_sort)

    def load_video(self) -> None:
        """
        Load the input video, and prepare the output video if required.
        """
        # load input video
        self.video = cv2.VideoCapture(self.video_path)
        if not self.video.isOpened():
            raise Exception("Error opening video file")

        # prepare output video writer if required
        if self.config["save_video"]:
            # read input video parameters
            frame_width = int(self.video.get(cv2.CAP_PROP_FRAME_WIDTH))
            frame_height = int(self.video.get(cv2.CAP_PROP_FRAME_HEIGHT))
            cap_fps = self.video.get(cv2.CAP_PROP_FPS)
            output_file = f"{self.video_file_root}_output_video.mp4"
            output_codec = cv2.VideoWriter_fourcc(*"H264")
            self.out = cv2.VideoWriter(
                output_file, output_codec, cap_fps, (frame_width, frame_height)
            )

    def prep_csv_writer(self) -> Tuple[Any, TextIO]:
        """
        Prepare csv writer to output tracking results
        """

        crabs_tracks_label_dir = Path("crabs_tracks_label")
        self.tracking_output_dir = (
            crabs_tracks_label_dir / self.video_file_root
        )
        # Create the subdirectory for the specific video file root
        self.tracking_output_dir.mkdir(parents=True, exist_ok=True)

        csv_file = open(
            f"{str(self.tracking_output_dir / self.video_file_root)}.csv",
            "w",
        )
        csv_writer = csv.writer(csv_file)

        # write header following VIA convention
        # https://www.robots.ox.ac.uk/~vgg/software/via/docs/face_track_annotation.html
        csv_writer.writerow(
            (
                "filename",
                "file_size",
                "file_attributes",
                "region_count",
                "region_id",
                "region_shape_attributes",
                "region_attributes",
            )
        )

        return csv_writer, csv_file

    def evaluate_tracking(
        self,
        gt_boxes_list: list,
        tracked_boxes_list: list,
    ) -> list[float]:
        """
        Evaluate tracking performance using the Multi-Object Tracking Accuracy (MOTA) metric.

        Parameters
        ----------
        gt_boxes_list : list[list[float]]
            List of ground truth bounding boxes for each frame.
        tracked_boxes_list : list[list[float]]
            List of tracked bounding boxes for each frame.

        Returns
        -------
        list[float]:
            The computed MOTA (Multi-Object Tracking Accuracy) score for the tracking performance.
        """
        mota_values = []
        prev_frame_ids: Optional[list[list[int]]] = None
        # prev_frame_ids = None
        for gt_boxes, tracked_boxes in zip(gt_boxes_list, tracked_boxes_list):
            mota = evaluate_mota(
                gt_boxes,
                tracked_boxes,
                self.config["iou_threshold"],
                prev_frame_ids,
            )
            mota_values.append(mota)
            # Update previous frame IDs for the next iteration
            prev_frame_ids = [[box[-1] for box in tracked_boxes]]

        return mota_values

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
        img = transform(frame).to(self.args.accelerator)
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
        pred_sort = self.prep_sort(prediction)
        tracked_boxes = self.sort_tracker.update(pred_sort)
        self.tracked_list.append(tracked_boxes)
        return tracked_boxes

    def save_required_output(
        self,
        tracked_boxes: list[list[float]],
        frame: np.ndarray,
        frame_number: int,
    ) -> None:
        """
        Handle the output based argument options.

        Parameters
        ----------
        tracked_boxes : list[list[float]]
            list of tracked bounding boxes.
        frame : np.ndarray
            The current frame.
        frame_number : int
            The frame number.
        """
        frame_name = f"{self.video_file_root}_frame_{frame_number:08d}.png"
        if self.config["save_csv_and_frames"]:
            save_frame_and_csv(
                frame_name,
                self.tracking_output_dir,
                tracked_boxes,
                frame,
                frame_number,
                self.csv_writer,
            )
        else:
            for bbox in tracked_boxes:
                write_tracked_bbox_to_csv(
                    bbox, frame, frame_name, self.csv_writer
                )

        if self.config["save_video"]:
            frame_copy = frame.copy()
            for bbox in tracked_boxes:
                xmin, ymin, xmax, ymax, id = bbox
                draw_bbox(
                    frame_copy,
                    (xmin, ymin),
                    (xmax, ymax),
                    (0, 0, 255),
                    f"id : {int(id)}",
                )
            self.out.write(frame_copy)

    def run_inference(self):
        """
        Run object detection + tracking on the video frames.
        """
        # initialisation
        frame_number = 1
        self.tracked_list = []

        self.csv_writer, csv_file = self.prep_csv_writer()

        # loop thru frames of clip
        while self.video.isOpened():
            # break if beyond end frame (mostly for debugging)
            if self.args.max_frames_to_read:
                if frame_number > self.args.max_frames_to_read:
                    break

            # read frame
            ret, frame = self.video.read()
            if not ret:
                print("No frame read. Exiting...")
                break

            prediction = self.get_prediction(frame)

            # run tracking
            self.prep_sort(prediction)
            tracked_boxes = self.update_tracking(prediction)
            self.save_required_output(tracked_boxes, frame, frame_number)

            # update frame
            frame_number += 1

        if self.args.gt_dir:
            gt_boxes_list = get_ground_truth_data(self.args.gt_dir)
            mota_values = self.evaluate_tracking(
                gt_boxes_list, self.tracked_list, self.config["iou_threshold"]
            )
            overall_mota = np.mean(mota_values)
            print("Overall MOTA:", overall_mota)

        # Close input video
        self.video.release()

        # Close outputs
        if self.config["save_video"]:
            self.out.release()

        if self.config["save_csv_and_frames"]:
            csv_file.close()


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

    inference = DetectorInference(args)
    inference.load_video()
    inference.run_inference()


def inference_parse_args(args):
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--checkpoint_path",
        type=str,
        required=True,
        help="location of checkpoint of the trained model",
    )
    parser.add_argument(
        "--video_path",
        type=str,
        required=True,
        help="location of images and coco annotation",
    )
    parser.add_argument(
        "--config_file",
        type=str,
        default=str(
            Path(__file__).parent / "config" / "inference_config.yaml"
        ),
        help=(
            "Location of YAML config to control training. "
            "Default: crabs-exploration/crabs/detection_tracking/config/inference_config.yaml"
        ),
    )
    parser.add_argument(
        "--output_path",
        type=str,
        default=os.getcwd(),
        help="location of output video",
    )
    parser.add_argument(
        "--accelerator",
        type=str,
        default="gpu",
        help="accelerator for pytorch lightning",
    )
    parser.add_argument(
        "--max_frames_to_read",
        type=int,
        default=None,
        help="Maximum number of frames to read (mostly for debugging).",
    )
    parser.add_argument(
        "--gt_dir",
        type=str,
        default=None,
        help="Location of json file containing ground truth annotations.",
    )
    args = parser.parse_args()
    main(args)


def app_wrapper():
    torch.set_float32_matmul_precision("medium")

    train_args = inference_parse_args(sys.argv[1:])
    main(train_args)


if __name__ == "__main__":
    app_wrapper()
