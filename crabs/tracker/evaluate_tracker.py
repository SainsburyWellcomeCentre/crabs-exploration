import logging
from typing import Any, Dict, Optional, Tuple

import numpy as np

from crabs.tracker.utils.tracking import calculate_iou


class TrackerEvaluate:
    def __init__(
        self,
        gt_dir: str,
        iou_threshold: float,
    ):
        """
        Initialize the TrackerEvaluate class with ground truth directory, tracked list, and IoU threshold.

        Parameters
        ----------
        gt_dir : str
            Directory path of the ground truth CSV file.
        iou_threshold : float
            Intersection over Union (IoU) threshold for evaluating tracking performance.
        """
        self.gt_dir = gt_dir
        self.iou_threshold = iou_threshold
        self.last_known_predicted_ids: Dict = {}

    def count_identity_switches(
        self,
        gt_to_tracked_id_previous_frame: Optional[Dict[int, int]],
        gt_to_tracked_id_current_frame: Dict[int, int],
    ) -> int:
        """
        Count the number of identity switches between two sets of object IDs.

        Parameters
        ----------
        gt_to_tracked_id_previous_frame : Optional[Dict[int, int]]
            A dictionary mapping ground truth IDs to predicted IDs from the previous frame.
        gt_to_tracked_id_current_frame : Dict[int, int]
            A dictionary mapping ground truth IDs to predicted IDs for the current frame.

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
        # Filter sets of ground truth IDs for current and previous frames to exclude NaN predicted IDs
        gt_ids_current_frame = set(gt_to_tracked_id_current_frame.keys())
        gt_ids_prev_frame = set(gt_to_tracked_id_previous_frame.keys())

        # Compute lists of ground truth IDs that continue, disappear, and appear
        gt_ids_cont = list(gt_ids_current_frame & gt_ids_prev_frame)
        gt_ids_disappear = list(gt_ids_prev_frame - gt_ids_current_frame)
        gt_ids_appear = list(gt_ids_current_frame - gt_ids_prev_frame)

        # Store used predicted IDs to avoid double counting
        # In `used_pred_ids` we log IDs from either the current or the previous frame that have been involved in an already counted ID switch.
        used_pred_ids = set()

        # Case 1: Objects that continue to exist according to GT
        for gt_id in gt_ids_cont:
            previous_pred_id = gt_to_tracked_id_previous_frame.get(gt_id)
            current_pred_id = gt_to_tracked_id_current_frame.get(gt_id)
            if all(
                not np.isnan(x) for x in [previous_pred_id, current_pred_id]
            ):  # Exclude if missed detection in previous AND current frame
                if current_pred_id != previous_pred_id:
                    switch_counter += 1
                    used_pred_ids.add(current_pred_id)
            # if the object was a missed detection in the previous frame: check if current prediction matches historical
            elif np.isnan(previous_pred_id) and not np.isnan(current_pred_id):
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
            if not np.isnan(
                previous_pred_id
            ):  # Exclude if missed detection in previous frame
                if previous_pred_id in gt_to_tracked_id_current_frame.values():
                    if previous_pred_id not in used_pred_ids:
                        switch_counter += 1
                        used_pred_ids.add(previous_pred_id)

        # Case 3: Objects that appear according to GT
        for gt_id in gt_ids_appear:
            current_pred_id = gt_to_tracked_id_current_frame.get(gt_id)
            if not np.isnan(
                current_pred_id
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

    def evaluate_mota(
        self,
        gt_data: Dict[str, np.ndarray],
        pred_data: Dict[str, np.ndarray],
        iou_threshold: float,
        gt_to_tracked_id_previous_frame: Optional[Dict[int, int]],
    ) -> Tuple[float, Dict[int, int]]:
        """
        Evaluate MOTA (Multiple Object Tracking Accuracy).

        Parameters
        ----------
        gt_data : Dict[str, np.ndarray]
            Dictionary containing ground truth bounding boxes and IDs.
            - 'bbox': Bounding boxes with shape (N, 4).
            - 'id': Ground truth IDs with shape (N,).
        pred_data : Dict[str, np.ndarray]
            Dictionary containing predicted bounding boxes and IDs.
            - 'bbox': Bounding boxes with shape (N, 4).
            - 'id': Predicted IDs with shape (N,).
        iou_threshold : float
            Intersection over Union (IoU) threshold for considering a match.
        gt_to_tracked_id_previous_frame : Optional[Dict[int, int]]
            A dictionary mapping ground truth IDs to predicted IDs from the previous frame.

        Returns
        -------
        float
            The computed MOTA (Multi-Object Tracking Accuracy) score for the tracking performance.
        Dict[int, int]
            A dictionary mapping ground truth IDs to predicted IDs for the current frame.
        """
        total_gt = len(gt_data["bbox"])
        false_positive = 0
        indices_of_matched_gt_boxes = set()
        gt_to_tracked_id_current_frame = {}

        pred_boxes = pred_data["bbox"]
        pred_ids = pred_data["id"]

        gt_boxes = gt_data["bbox"]
        gt_ids = gt_data["id"]

        for i, (pred_box, pred_id) in enumerate(zip(pred_boxes, pred_ids)):
            best_iou = 0.0
            index_gt_best_match = None
            index_gt_not_match = None

            for j, gt_box in enumerate(gt_boxes):
                if j not in indices_of_matched_gt_boxes:
                    iou = calculate_iou(gt_box, pred_box)
                    if iou > iou_threshold and iou > best_iou:
                        best_iou = iou
                        index_gt_best_match = j
                    else:
                        index_gt_not_match = j

            if index_gt_best_match is not None:
                # Successfully found a matching ground truth box for the tracked box.
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
                ] = np.nan

        missed_detections = total_gt - len(indices_of_matched_gt_boxes)
        num_switches = self.count_identity_switches(
            gt_to_tracked_id_previous_frame, gt_to_tracked_id_current_frame
        )

        mota = (
            1 - (missed_detections + false_positive + num_switches) / total_gt
        )

        return mota, gt_to_tracked_id_current_frame

    def evaluate_tracking(
        self,
        ground_truth_dict: Dict[int, Dict[str, Any]],
        predicted_dict: Dict[int, Dict[str, Any]],
    ) -> list[float]:
        """
        Evaluate tracking performance using the Multi-Object Tracking Accuracy (MOTA) metric.

        Parameters
        ----------
        ground_truth_dict : dict
            Dictionary containing ground truth bounding boxes and IDs for each frame, organized by frame number.
        predicted_dict : dict
            Dictionary containing predicted bounding boxes and IDs for each frame, organized by frame number.

        Returns
        -------
        list[float]:
            The computed MOTA (Multi-Object Tracking Accuracy) score for the tracking performance.
        """
        mota_values = []
        prev_frame_id_map: Optional[dict] = None

        for frame_number in sorted(ground_truth_dict.keys()):
            gt_data_frame = ground_truth_dict[frame_number]

            if frame_number < len(predicted_dict):
                pred_data_frame = predicted_dict[frame_number]
                mota, prev_frame_id_map = self.evaluate_mota(
                    gt_data_frame,
                    pred_data_frame,
                    self.iou_threshold,
                    prev_frame_id_map,
                )
                mota_values.append(mota)

        return mota_values

    def run_evaluation(self, predicted_dict, ground_truth_dict) -> None:
        """
        Run evaluation of tracking based on tracking ground truth.
        """
        mota_values = self.evaluate_tracking(ground_truth_dict, predicted_dict)

        overall_mota = np.mean(mota_values)
        logging.info("Overall MOTA: %f" % overall_mota)
