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
        gt_dir: str,  # annotations_file
        predicted_boxes_dict: dict,
        iou_threshold: float,
        tracking_output_dir: Path,
    ):
        """Initialize the TrackerEvaluate class.

        Initialised with ground truth directory, tracked list, and IoU
        threshold.

        Parameters
        ----------
        gt_dir : str
            Directory path of the ground truth CSV file.
        predicted_boxes_dict : dict
            Dictionary mapping frame indices to bounding boxes arrays
            (under "bboxes_tracked") and bounding boxes scores (under
            "bboxes_scores"). The bounding boxes array have shape
            (n, 5) where n is the number of boxes in the frame and
            the 5 columns are (xmin, ymin, xmax, ymax, id).
        iou_threshold : float
            Intersection over Union (IoU) threshold for evaluating
            tracking performance.
        tracking_output_dir : Path
            Path to the directory where the tracking output will be saved.

        """
        self.gt_dir = gt_dir
        self.predicted_boxes_dict = predicted_boxes_dict
        self.iou_threshold = iou_threshold
        self.tracking_output_dir = tracking_output_dir
        self.last_known_predicted_ids: dict = {}

    def get_predicted_data(self) -> dict[int, dict[str, Any]]:
        """Format predicted bounding box and ID as dictionary.

        Dictionary keys are frame numbers. It splits bounding boxes
        array of input dictionary.

        Returns
        -------
        dict[int, dict[str, Any]]:
            A dictionary where the key is the frame number and the value is
            another dictionary containing:
            - 'bbox': A numpy array with shape (N, 4) containing coordinates
            of the bounding boxes [x, y, x + width, y + height] for every
            object in the frame.
            - 'id': A numpy array containing the IDs of the tracked objects.

        """
        # TODO: we probably can do away with this function and
        # just use "predicted_boxes_dict" directly
        predicted_dict: dict[int, dict[str, Any]] = {}

        for frame_idx in self.predicted_boxes_dict:
            predicted_bboxes_array = self.predicted_boxes_dict[frame_idx][
                "bboxes_tracked"
            ]

            if predicted_bboxes_array.size == 0:  # why? no predictions?
                continue

            bboxes = predicted_bboxes_array[:, :4]
            ids = predicted_bboxes_array[:, 4]

            predicted_dict[frame_idx] = {"bbox": bboxes, "id": ids}

        return predicted_dict

    def get_ground_truth_data(self) -> dict[int, dict[str, Any]]:
        """Fromat ground truth bounding box data as dict with key frame number.

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

        with open(self.gt_dir) as csvfile:
            csvreader = csv.reader(csvfile)
            next(csvreader)  # Skip the header row
            ground_truth_data = [
                extract_bounding_box_info(row) for row in csvreader
            ]

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

        # format as numpy arrays
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
        if gt_to_tracked_id_previous_frame is None:
            for gt_id, pred_id in gt_to_tracked_id_current_frame.items():
                if not np.isnan(pred_id):
                    self.last_known_predicted_ids[gt_id] = pred_id
            return 0

        switch_counter = 0
        # Filter sets of ground truth IDs for current and previous frames
        # to exclude NaN predicted IDs
        gt_ids_current_frame = set(gt_to_tracked_id_current_frame.keys())
        gt_ids_prev_frame = set(gt_to_tracked_id_previous_frame.keys())

        # Compute lists of ground truth IDs that continue, disappear,
        # and appear
        gt_ids_cont = list(gt_ids_current_frame & gt_ids_prev_frame)
        gt_ids_disappear = list(gt_ids_prev_frame - gt_ids_current_frame)
        gt_ids_appear = list(gt_ids_current_frame - gt_ids_prev_frame)

        # Store used predicted IDs to avoid double counting
        # In `used_pred_ids` we log IDs from either the current or the
        # previous frame that have been involved in an already
        # counted ID switch.
        used_pred_ids = set()

        # Case 1: Objects that continue to exist according to GT
        for gt_id in gt_ids_cont:
            previous_pred_id = gt_to_tracked_id_previous_frame.get(gt_id)
            current_pred_id = gt_to_tracked_id_current_frame.get(gt_id)
            if all(
                not np.isnan(x)  # type: ignore
                for x in [previous_pred_id, current_pred_id]
            ):  # Exclude if missed detection in previous AND current frame
                if current_pred_id != previous_pred_id:
                    switch_counter += 1
                    used_pred_ids.add(current_pred_id)
            # if the object was a missed detection in the previous frame:
            # check if current prediction matches historical
            elif np.isnan(previous_pred_id) and not np.isnan(current_pred_id):  # type: ignore  # noqa: SIM102
                if gt_id in self.last_known_predicted_ids:
                    last_known_predicted_id = self.last_known_predicted_ids[
                        gt_id
                    ]
                    if current_pred_id != last_known_predicted_id:
                        switch_counter += 1
            # save most recent predicted ID associated to this groundtruth ID
            self.last_known_predicted_ids[gt_id] = current_pred_id

        # Case 2: Objects that disappear according to GT
        for gt_id in gt_ids_disappear:
            previous_pred_id = gt_to_tracked_id_previous_frame.get(gt_id)
            if not np.isnan(  # noqa: SIM102
                previous_pred_id  # type: ignore
            ):  # Exclude if missed detection in previous frame
                if previous_pred_id in gt_to_tracked_id_current_frame.values():  # noqa: SIM102
                    if previous_pred_id not in used_pred_ids:
                        switch_counter += 1
                        used_pred_ids.add(previous_pred_id)

        # Case 3: Objects that appear according to GT
        for gt_id in gt_ids_appear:
            current_pred_id = gt_to_tracked_id_current_frame.get(gt_id)
            if not np.isnan(
                current_pred_id  # type: ignore
            ):  # Exclude if missed detection in current frame
                # check if there was and ID switch wrt previous frame
                if current_pred_id in gt_to_tracked_id_previous_frame.values():
                    if previous_pred_id not in used_pred_ids:
                        switch_counter += 1
                # if ID not immediately swapped from previous frame:
                # check if predicted ID matches the last known one
                elif gt_id in self.last_known_predicted_ids:
                    last_known_predicted_id = self.last_known_predicted_ids[
                        gt_id
                    ]
                    if current_pred_id != last_known_predicted_id:
                        switch_counter += 1
                self.last_known_predicted_ids[gt_id] = current_pred_id

        return switch_counter

    def compute_mota_one_frame(
        self,
        gt_data: dict[str, np.ndarray],
        pred_data: dict[str, np.ndarray],
        iou_threshold: float,
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
        iou_threshold : float
            Intersection over Union (IoU) threshold for considering a match.
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
        false_positive = 0
        true_positive = 0
        indices_of_matched_gt_boxes = set()
        gt_to_tracked_id_current_frame = {}

        pred_boxes = pred_data["bbox"]
        pred_ids = pred_data["id"]

        gt_boxes = gt_data["bbox"]
        gt_ids = gt_data["id"]

        for _i, (pred_box, pred_id) in enumerate(zip(pred_boxes, pred_ids)):
            best_iou = 0.0
            index_gt_best_match = None
            index_gt_not_match = None

            for j, gt_box in enumerate(gt_boxes):
                if j not in indices_of_matched_gt_boxes:
                    iou = self.calculate_iou(gt_box, pred_box)
                    if iou > iou_threshold and iou > best_iou:
                        best_iou = iou
                        index_gt_best_match = j
                    else:
                        index_gt_not_match = j

            if index_gt_best_match is not None:
                true_positive += 1
                # Successfully found a matching ground truth box for the
                # tracked box.
                indices_of_matched_gt_boxes.add(index_gt_best_match)
                # Map ground truth ID to tracked ID
                gt_to_tracked_id_current_frame[
                    int(gt_ids[index_gt_best_match])
                ] = int(pred_id)
            else:
                false_positive += 1
            if index_gt_not_match is not None:
                gt_to_tracked_id_current_frame[
                    int(gt_ids[index_gt_not_match])
                ] = np.nan  # type: ignore

        missed_detections = total_gt - len(indices_of_matched_gt_boxes)
        num_switches = self.count_identity_switches(
            gt_to_tracked_id_previous_frame, gt_to_tracked_id_current_frame
        )

        mota = (
            1 - (missed_detections + false_positive + num_switches) / total_gt
        )
        return (
            mota,
            true_positive,
            missed_detections,
            false_positive,
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
            frame, organized by frame number.
        predicted_dict : dict
            Dictionary containing predicted bounding boxes and IDs for each
            frame, organized by frame _index_.

        Returns
        -------
        list[float]:
            The computed MOTA (Multi-Object Tracking Accuracy) score for the
            tracking performance.

        """
        mota_values = []
        prev_frame_id_map: Optional[dict] = None
        results: dict[str, Any] = {
            "Frame Number": [],
            "Total Ground Truth": [],
            "True Positives": [],
            "Missed Detections": [],
            "False Positives": [],
            "Number of Switches": [],
            "MOTA": [],
        }

        for frame_number in sorted(ground_truth_dict.keys()):
            gt_data_frame = ground_truth_dict[frame_number]

            if frame_number < len(predicted_dict):
                pred_data_frame = predicted_dict[frame_number]

                (
                    mota,
                    true_positives,
                    missed_detections,
                    false_positives,
                    num_switches,
                    total_gt,
                    prev_frame_id_map,
                ) = self.compute_mota_one_frame(
                    gt_data_frame,
                    pred_data_frame,
                    self.iou_threshold,
                    prev_frame_id_map,
                )
                mota_values.append(mota)
                results["Frame Number"].append(frame_number)
                results["Total Ground Truth"].append(total_gt)
                results["True Positives"].append(true_positives)
                results["Missed Detections"].append(missed_detections)
                results["False Positives"].append(false_positives)
                results["Number of Switches"].append(num_switches)
                results["MOTA"].append(mota)

        save_tracking_mota_metrics(self.tracking_output_dir, results)

        return mota_values

    def run_evaluation(self) -> None:
        """Run evaluation of tracking based on tracking ground truth."""
        predicted_dict = self.get_predicted_data()
        ground_truth_dict = self.get_ground_truth_data()
        mota_values = self.evaluate_tracking(ground_truth_dict, predicted_dict)

        overall_mota = np.mean(mota_values)
        logging.info("Overall MOTA: %f" % overall_mota)  # noqa: UP031
