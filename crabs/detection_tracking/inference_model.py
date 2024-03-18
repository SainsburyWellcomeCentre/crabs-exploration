import argparse
import csv
import os
from pathlib import Path
from typing import Any, List, Optional, TextIO, Tuple, Union

import cv2
import numpy as np
import torch
import torchvision.transforms as transforms
from sort import Sort

from crabs.detection_tracking.detection_utils import (
    draw_bbox,
    save_frame_and_csv,
)


class DetectorInference:
    """
    A class for performing object detection or tracking inference on a video
    using a trained model.

    Parameters:
    ----------
    args : argparse.Namespace)
        Command-line arguments containing configuration settings.

    Attributes:
    ----------
    args : argparse.Namespace
        The command-line arguments provided.
    vid_path : str
        The path to the input video.
    iou_threshold : float
        The iou threshold for tracking.
    score_threshold : float
        The score confidence threshold for tracking.
    sort_tracker : Sort
        An instance of the sorting algorithm used for tracking.
    """

    def __init__(self, args: argparse.Namespace) -> None:
        self.args = args
        self.vid_path = args.vid_path
        self.score_threshold = args.score_threshold
        self.iou_threshold = args.iou_threshold
        self.sort_tracker = Sort(
            max_age=args.max_age,
            min_hits=args.min_hits,
            iou_threshold=self.iou_threshold,
        )
        self.video_file_root = f"{Path(self.vid_path).stem}"

    def load_trained_model(self) -> torch.nn.Module:
        """
        Load the trained model.

        Returns
        -------
        torch.nn.Module
        """
        model = torch.load(
            self.args.model_dir,
            map_location=torch.device(self.args.accelerator),
        )
        model.eval()
        return model

    def prep_sort(self, prediction: dict) -> np.ndarray:
        """
        Put predictions in format expected by SORT

        Parameters:
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
            if score > self.score_threshold:
                bbox = np.concatenate((box, [score]))
                pred_sort.append(bbox)

        return np.asarray(pred_sort)

    def load_video(self) -> None:
        """
        Load the input video, and prepare the output video if required.
        """
        # load trained model
        self.trained_model = self.load_trained_model()

        # load input video
        self.video = cv2.VideoCapture(self.vid_path)
        if not self.video.isOpened():
            raise Exception("Error opening video file")

        # prepare output video writer if required
        if self.args.save_video:
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

        self.tracking_output_dir = (
            Path("crabs_tracks_label") / self.video_file_root
        )
        crabs_tracks_label_dir = Path("crabs_tracks_label")
        crabs_tracks_label_dir.mkdir(parents=True, exist_ok=True)

        # Create the subdirectory for the specific video file root
        self.tracking_output_dir.mkdir(parents=True, exist_ok=True)

        csv_file = open(
            f"{str(self.tracking_output_dir)}/{str(self.video_file_root)}_0.csv",
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
        iou_threshold: float,
    ) -> List[float]:
        """
        Evaluate tracking performance using the Multi-Object Tracking Accuracy (MOTA) metric.

        Parameters:
        -----------
        gt_boxes_list : List[list]
            List of ground truth bounding boxes for each frame.
        tracked_boxes_list : List[list]
            List of tracked bounding boxes for each frame.
        iou_threshold : float
            The IoU threshold used to determine matches between ground truth and tracked boxes.

        Returns:
        --------
        float:
            The computed MOTA (Multi-Object Tracking Accuracy) score for the tracking performance.
        """
        from crabs.detection_tracking.detection_utils import evaluate_mota

        mota_values = []
        prev_frame: Union[Optional[List[List[float]]], None] = None
        for gt_boxes, tracked_boxes in zip(gt_boxes_list, tracked_boxes_list):
            mota = evaluate_mota(
                gt_boxes, tracked_boxes, iou_threshold, prev_frame
            )
            mota_values.append(mota)
            # Update previous frame IDs for the next iteration
            prev_frame = [list(box) for box in tracked_boxes]

        return mota_values

    def get_prediction(self, frame: np.ndarray) -> torch.Tensor:
        """
        Get prediction from the trained model for a given frame.

        Parameters:
        -----------
        frame : np.ndarray
            The input frame for which prediction is to be obtained.

        Returns:
        --------
        torch.Tensor:
            The prediction tensor from the trained model.
        """
        transform = transforms.Compose([transforms.ToTensor()])
        img = transform(frame).to(self.args.accelerator)
        img = img.unsqueeze(0)
        return self.trained_model(img)

    def update_tracking(self, prediction: dict) -> List[List[float]]:
        """
        Update the tracking system with the latest prediction.

        Parameters:
        -----------
        prediction : dict
            Dictionary containing predicted bounding boxes, scores, and labels.

        Returns:
        --------
        List[List[float]]:
            List of tracked bounding boxes after updating the tracking system.
        """
        pred_sort = self.prep_sort(prediction)
        tracked_boxes = self.sort_tracker.update(pred_sort)
        self.tracked_list.append(tracked_boxes)
        return tracked_boxes

    def handle_output(
        self,
        tracked_boxes: List[List[float]],
        frame: np.ndarray,
        frame_number: int,
    ) -> None:
        """
        Handle the output based argument options.

        Parameters:
        -----------
        tracked_boxes : List[List[float]]
            List of tracked bounding boxes.
        frame : np.ndarray
            The current frame.
        frame_number : int
            The frame number.
        """
        if self.args.save_csv_and_frames:
            if self.args.save_video:
                frame_copy = save_frame_and_csv(
                    self.video_file_root,
                    self.tracking_output_dir,
                    tracked_boxes,
                    frame,
                    frame_number,
                    self.csv_writer,
                )
                self.out.write(frame_copy)
            else:
                save_frame_and_csv(
                    self.video_file_root,
                    self.tracking_output_dir,
                    tracked_boxes,
                    frame,
                    frame_number,
                    self.csv_writer,
                    save_plot=False,
                )
        elif self.args.save_video:
            frame_copy = frame.copy()
            for bbox in tracked_boxes:
                xmin, ymin, xmax, ymax, id = bbox
                draw_bbox(
                    frame_copy,
                    int(xmin),
                    int(ymin),
                    int(xmax),
                    int(ymax),
                    (0, 0, 255),
                    f"id : {int(id)}",
                )
            self.out.write(frame)

    def run_inference(self):
        """
        Run object detection + tracking on the video frames.
        """
        # Get transform to tensor
        transforms.Compose([transforms.ToTensor()])

        # initialisation
        frame_number = 1
        self.tracked_list = []

        # initialise csv writer if required
        if self.args.save_csv_and_frames:
            self.csv_writer, csv_file = self.prep_csv_writer()

        if self.args.gt_dir:
            from crabs.detection_tracking.detection_utils import (
                get_ground_truth_data,
            )

            gt_boxes_list = get_ground_truth_data(self.args.gt_dir)

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
            self.handle_output(tracked_boxes, frame, frame_number)

            # update frame
            frame_number += 1

        if self.args.gt_dir:
            mota_values = self.evaluate_tracking(
                gt_boxes_list, self.tracked_list, self.iou_threshold
            )
            overall_mota = np.mean(mota_values)
            print("Overall MOTA:", overall_mota)

        # # Close input video
        self.video.release()

        # Close outputs
        if self.args.save_video:
            self.out.release()

        if args.save_csv_and_frames:
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


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model_dir",
        type=str,
        required=True,
        help="location of trained model",
    )
    parser.add_argument(
        "--vid_path",
        type=str,
        required=True,
        help="location of images and coco annotation",
    )
    parser.add_argument(
        "--save_video",
        action="store_true",
        help="save video inference",
    )
    parser.add_argument(
        "--output_path",
        type=str,
        default=os.getcwd(),
        help="location of output video",
    )
    parser.add_argument(
        "--score_threshold",
        type=float,
        default=0.1,
        help="threshold for prediction score",
    )
    parser.add_argument(
        "--iou_threshold",
        type=float,
        default=0.1,
        help="threshold for prediction score",
    )
    parser.add_argument(
        "--max_age",
        type=int,
        default=10,
        help="Maximum number of frames to keep alive a track without associated detections.",
    )
    parser.add_argument(
        "--min_hits",
        type=int,
        default=1,
        help="Minimum number of associated detections before track is initialised.",
    )
    parser.add_argument(
        "--accelerator",
        type=str,
        default="gpu",
        help="accelerator for pytorch lightning",
    )
    parser.add_argument(
        "--save_csv_and_frames",
        action="store_true",
        help=(
            "Save predicted tracks in VIA csv format and export corresponding frames. "
            "This is useful to prepare for manual labelling of tracks."
        ),
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
        help="Directory contains ground truth annotations.",
    )
    args = parser.parse_args()
    main(args)
