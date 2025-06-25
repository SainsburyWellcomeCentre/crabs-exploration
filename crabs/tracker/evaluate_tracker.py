"""Evaluate tracker using the Multi-Object Tracking Accuracy (MOTA) metric."""

import csv
import logging
from pathlib import Path
from typing import Any, Optional

import numpy as np

from crabs.tracker.utils.tracking import (
    extract_bounding_box_info,
    save_tracking_mota_metrics,
)


class TrackerEvaluate:
    """Interface to evaluate tracker."""

    def __init__(
        self,
        input_video_file_root: str,
        annotations_file: str,
        predicted_boxes_dict: dict,
        iou_threshold: float,
        tracking_output_dir: Path,
    ):
        """Initialize the TrackerEvaluate class.

        Initialised with ground truth directory, tracked list, and IoU
        threshold.

        Parameters
        ----------
        input_video_file_root : str
            Filename without extension to the input video file.
        annotations_file : str
            Path to the ground truth annotations CSV file.
        predicted_boxes_dict : dict
            Dictionary mapping frame indices to bounding boxes arrays
            (under "tracked_boxes"), ids (under "ids") and detection scores
            (under "scores"). The bounding boxes array have shape (n, 4) where
            n is the number of boxes in the frame and the 4 columns are (xmin,
            ymin, xmax, ymax).
        iou_threshold : float
            Intersection over Union (IoU) threshold used to evaluate
            tracking performance.
        tracking_output_dir : Path
            Path to the directory where the tracking output will be saved.

        """
        self.input_video_file_root = input_video_file_root
        self.annotations_file = annotations_file
        self.predicted_boxes_dict = predicted_boxes_dict
        self.iou_threshold = iou_threshold
        self.tracking_output_dir = tracking_output_dir
        self.last_known_predicted_ids: dict = {}

    def get_ground_truth_data(self) -> dict[int, dict[str, Any]]:
        """Fromat ground truth data as a dictionary with key frame number.

        Returns
        -------
        dict[int, dict[str, Any]]:
            A dictionary where the key is the frame number and the value is
            another dictionary containing:
            - 'bbox': A numpy arrays with shape of (N, 4) containing
                coordinates of the bounding box [x, y, x + width, y + height]
                for every crabs in the frame.
            - 'id': The ground truth ID

        """
        # TODO: refactor with pandas
        with open(self.annotations_file) as csvfile:
            csvreader = csv.reader(csvfile)
            next(csvreader)  # Skip the header row
            ground_truth_data = [
                extract_bounding_box_info(row) for row in csvreader
            ]  # assumes frame number is the last part of the filename
            # and is preceded by an underscore

        # Format as a dictionary with key = frame number
        ground_truth_dict: dict = {}

        # loop thru annotations
        for data in ground_truth_data:
            # Get frame, bbox, id
            frame_number = data["frame_number"]
            bbox = np.array(
                [
                    data["x"],
                    data["y"],
                    data["x"] + data["width"],
                    data["y"] + data["height"],
                ],
                dtype=np.float32,
            )
            track_id = int(float(data["id"]))

            # If frame does not exist in dict: initialise
            if frame_number not in ground_truth_dict:
                ground_truth_dict[frame_number] = {"bbox": [], "id": []}

            # Append bbox and id to the dictionary
            ground_truth_dict[frame_number]["bbox"].append(bbox)
            ground_truth_dict[frame_number]["id"].append(track_id)

        # Format bbox and id as numpy arrays per frame
        for frame_number in ground_truth_dict:
            ground_truth_dict[frame_number]["bbox"] = np.array(
                ground_truth_dict[frame_number]["bbox"], dtype=np.float32
            )
            ground_truth_dict[frame_number]["id"] = np.array(
                ground_truth_dict[frame_number]["id"], dtype=np.float32
            )
        return ground_truth_dict

    def calculate_iou(self, box1: np.ndarray, box2: np.ndarray) -> float:
        """Calculate IoU (Intersection over Union) of two bounding boxes.

        Parameters
        ----------
        box1 : np.ndarray
            Coordinates [x1, y1, x2, y2] of the first bounding box.
            Here, (x1, y1) represents the top-left corner, and (x2, y2)
            represents the bottom-right corner.
        box2 : np.ndarray
            Coordinates [x1, y1, x2, y2] of the second bounding box.
            Here, (x1, y1) represents the top-left corner, and (x2, y2)
            represents the bottom-right corner.

        Returns
        -------
        float
            IoU value.

        """
        x1_box1, y1_box1, x2_box1, y2_box1 = box1
        x1_box2, y1_box2, x2_box2, y2_box2 = box2

        # Calculate intersection coordinates
        # x1, y1 intersect is the x1,y1 corner closer to the image's
        # bottom-right corner
        # x2, y2 intersect is the x2,y2 corner closer to the image's
        # top-left corner
        x1_intersect = max(x1_box1, x1_box2)
        y1_intersect = max(y1_box1, y1_box2)
        x2_intersect = min(x2_box1, x2_box2)
        y2_intersect = min(y2_box1, y2_box2)

        # Calculate area of intersection rectangle
        intersect_width = max(0, x2_intersect - x1_intersect + 1)
        intersect_height = max(0, y2_intersect - y1_intersect + 1)
        intersect_area = intersect_width * intersect_height

        # Calculate area of individual bounding boxes
        box1_area = (x2_box1 - x1_box1 + 1) * (y2_box1 - y1_box1 + 1)
        box2_area = (x2_box2 - x1_box2 + 1) * (y2_box2 - y1_box2 + 1)

        iou = intersect_area / float(box1_area + box2_area - intersect_area)

        return iou

    def count_identity_switches(  # noqa: C901
        self,
        gt_to_tracked_id_previous_frame: Optional[dict[int, int]],
        gt_to_tracked_id_current_frame: dict[int, int],
    ) -> int:
        """Count the number of identity switches between two sets of IDs.

        Parameters
        ----------
        gt_to_tracked_id_previous_frame : Optional[dict[int, int]]
            A dictionary mapping ground truth IDs to predicted IDs from the
            previous frame.
        gt_to_tracked_id_current_frame : dict[int, int]
            A dictionary mapping ground truth IDs to predicted IDs for the
            current frame.

        Returns
        -------
        int
            The number of identity switches between the two sets of object IDs.

        """
        switch_counter = 0

        if not gt_to_tracked_id_previous_frame:
            return switch_counter

        # Count cases a current GT ID maps to different predicted IDs
        # in the current and previous frame (ignoring nan predicted IDs)
        for gt_id in gt_to_tracked_id_current_frame:
            pred_id_current_frame = gt_to_tracked_id_current_frame[gt_id]
            pred_id_previous_frame = gt_to_tracked_id_previous_frame.get(
                gt_id, np.nan
            )
            if (
                not np.isnan(pred_id_current_frame)
                and not np.isnan(pred_id_previous_frame)
                and pred_id_current_frame != pred_id_previous_frame
            ):
                switch_counter += 1

        # Count cases a current predicted ID maps to different GT IDs
        # in the current and previous frame (ignoring nan predicted IDs)
        for pred_id_current_frame in gt_to_tracked_id_current_frame.values():
            if (
                not np.isnan(pred_id_current_frame)
                and pred_id_current_frame in gt_to_tracked_id_previous_frame.values()
            ):
                # Get corresponding GT ID from current frame
                gt_id_current_frame = [
                    ky
                    for ky, val in gt_to_tracked_id_current_frame.items()
                    if val == pred_id_current_frame
                ][0]

                # Get corresponding GT ID from previous frame
                gt_id_previous_frame = [
                    ky
                    for ky, val in gt_to_tracked_id_previous_frame.items()
                    if val == pred_id_current_frame
                ][0]

                # Check if they match
                if gt_id_current_frame != gt_id_previous_frame:
                    switch_counter += 1

        return switch_counter

    def compute_mota_one_frame(
        self,
        gt_data: dict[str, np.ndarray],
        pred_data: dict[str, np.ndarray],
        gt_to_tracked_id_previous_frame: Optional[dict[int, int]],
    ) -> tuple[float, int, int, int, int, int, dict[int, int]]:
        """Evaluate MOTA (Multiple Object Tracking Accuracy).

        Parameters
        ----------
        gt_data : dict[str, np.ndarray]
            Dictionary containing ground truth bounding boxes and IDs.
            - 'bbox': Bounding boxes with shape (N, 4).
            - 'id': Ground truth IDs with shape (N,).
        pred_data : dict[str, np.ndarray]
            Dictionary containing predicted bounding boxes and IDs.
            - 'bbox': Bounding boxes with shape (N, 4).
            - 'id': Predicted IDs with shape (N,).
        gt_to_tracked_id_previous_frame : Optional[dict[int, int]]
            A dictionary mapping ground truth IDs to predicted IDs from the
            previous frame.

        Returns
        -------
        float
            The computed MOTA (Multi-Object Tracking Accuracy) score for the
            tracking performance.
        dict[int, int]
            A dictionary mapping ground truth IDs to predicted IDs for the
            current frame.

        """
        total_gt = len(gt_data["bbox"])
        false_positives = 0
        true_positives = 0
        indices_of_matched_gt_boxes = set()

        pred_boxes = pred_data["tracked_boxes"]
        pred_ids = pred_data["ids"]

        gt_boxes = gt_data["bbox"]
        gt_ids = gt_data["id"]

        # Initialise dictionary to map ground truth IDs to tracked IDs
        gt_to_tracked_id_current_frame = {
            gt_id: np.nan for gt_id in gt_data["id"]
        }

        # Loop through detections
        for pred_box, pred_id in zip(pred_boxes, pred_ids):
            best_iou = 0.0
            index_gt_best_match = None

            # Look for best matching ground truth box
            for j, gt_box in enumerate(gt_boxes):
                if j not in indices_of_matched_gt_boxes:
                    iou = self.calculate_iou(gt_box, pred_box)
                    if iou > self.iou_threshold and iou > best_iou:
                        best_iou = iou
                        index_gt_best_match = j

            # If no best match is found, add to false positives
            if index_gt_best_match is None:
                false_positives += 1
            # If a best match is found, add to true positives
            else:
                true_positives += 1

                # Log index of best match
                indices_of_matched_gt_boxes.add(index_gt_best_match)

                # Overwrite ground truth ID with matched tracked ID
                gt_to_tracked_id_current_frame[
                    int(gt_ids[index_gt_best_match])
                ] = int(pred_id)

        # Count missed detections
        missed_detections = total_gt - len(indices_of_matched_gt_boxes)

        # Count identity switches
        num_switches = self.count_identity_switches(
            gt_to_tracked_id_previous_frame,
            gt_to_tracked_id_current_frame,
        )

        # Compute MOTA
        mota = (
            1 - (missed_detections + false_positives + num_switches) / total_gt
        )
        return (
            mota,
            true_positives,
            missed_detections,
            false_positives,
            num_switches,
            total_gt,
            gt_to_tracked_id_current_frame,
        )

    def evaluate_tracking(
        self,
        ground_truth_dict: dict[int, dict[str, Any]],
        predicted_dict: dict[int, dict[str, Any]],
    ) -> list[float]:
        """Evaluate tracking with the Multi-Object Tracking Accuracy metric.

        Parameters
        ----------
        ground_truth_dict : dict
            Dictionary containing ground truth bounding boxes and IDs for each
            frame, keyed by frame number.
        predicted_dict : dict
            Dictionary containing predicted bounding boxes and IDs for each
            frame, keyed by frame index.

        Returns
        -------
        list[float]:
            The computed MOTA (Multi-Object Tracking Accuracy) score for the
            tracking performance.

        """
        # Initialise output variables
        results: dict[str, Any] = {
            "Frame Number": [],
            "Frame Index": [],
            "Total Ground Truth": [],
            "True Positives": [],
            "Missed Detections": [],
            "False Positives": [],
            "Number of Switches": [],
            "MOTA": [],
        }

        # Initialise previous frame ID map to track ID switches
        gt_to_tracked_id_previous_frame: Optional[dict] = None

        # Check that the number of frames in the ground truth and predictions
        # are the same, print warning if not
        if len(ground_truth_dict) > len(predicted_dict):
            logging.warning(
                "There are more frames in the ground truth than in the "
                "predictions."
                f"Only the first {len(predicted_dict)} frames will be "
                "evaluated."
                "To match the frames across ground truth and predictions, "
                "we assume the first frame number in the ground truth is the "
                "frame with index 0 in the predictions."
            )
        elif len(ground_truth_dict) < len(predicted_dict):
            logging.warning(
                "There are more frames in the predictions than in the "
                "ground truth."
                f"Only the first {len(ground_truth_dict)} frames will be "
                "evaluated."
                "To match the frames across ground truth and predictions, "
                "we assume the first frame number in the ground truth is the "
                "frame with index 0 in the predictions."
            )

        # Loop through frame numbers in ground truth dict
        # We assume the first frame number in the ground truth is the frame
        # with index 0 in the predictions dict
        start_frame_number = min(ground_truth_dict.keys())
        for frame_number in ground_truth_dict:
            # Infer frame index from frame number
            frame_index = frame_number - start_frame_number

            # Skip if the frame index is not in the predictions dict
            if frame_index not in predicted_dict:
                logging.warning(
                    f"Frame {frame_number} is not in the predictions."
                    "Skipping evaluation."
                )
                continue

            # Compute MOTA for the frame
            (
                mota,
                true_positives,
                missed_detections,
                false_positives,
                num_switches,
                total_gt,
                gt_to_tracked_id_previous_frame,
            ) = self.compute_mota_one_frame(
                ground_truth_dict[frame_number],
                predicted_dict[frame_index],
                gt_to_tracked_id_previous_frame,
            )

            # Append results
            results["Frame Number"].append(frame_number)
            results["Frame Index"].append(frame_index)
            results["Total Ground Truth"].append(total_gt)
            results["True Positives"].append(true_positives)
            results["Missed Detections"].append(missed_detections)
            results["False Positives"].append(false_positives)
            results["Number of Switches"].append(num_switches)
            results["MOTA"].append(mota)

        # Save results to CSV file
        save_tracking_mota_metrics(
            self.tracking_output_dir,
            self.input_video_file_root,
            results,
        )

        return results["MOTA"]

    def run_evaluation(self) -> None:
        """Run evaluation of tracking based on tracking ground truth."""
        ground_truth_dict = self.get_ground_truth_data()
        mota_values = self.evaluate_tracking(
            ground_truth_dict,
            self.predicted_boxes_dict,
        )

        overall_mota = np.mean(mota_values)
        logging.info(f"Mean MOTA over all frames: {overall_mota}")
