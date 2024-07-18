import csv
import logging
from typing import Any, Dict, Optional, Tuple

import numpy as np

from crabs.tracker.utils.tracking import extract_bounding_box_info


class TrackerEvaluate:
    def __init__(
        self,
        gt_dir: str,
        predicted_boxes_id: list[np.ndarray],
        iou_threshold: float,
    ):
        """
        Initialize the TrackerEvaluate class with ground truth directory, tracked list, and IoU threshold.

        Parameters
        ----------
        gt_dir : str
            Directory path of the ground truth CSV file.
        tracked_list : List[np.ndarray]
            A list where each element is a numpy array representing tracked objects in a frame.
            Each numpy array has shape (N, 5), where N is the number of objects.
            The columns are [x1, y1, x2, y2, id], where (x1, y1) and (x2, y2)
            define the bounding box and id is the object ID.
        iou_threshold : float
            Intersection over Union (IoU) threshold for evaluating tracking performance.
        """
        self.gt_dir = gt_dir
        self.predicted_boxes_id = predicted_boxes_id
        self.iou_threshold = iou_threshold
        self.last_known_predicted_ids: Dict = {}
        self.total_num_switches = 0

    def get_predicted_data(self) -> Dict[int, Dict[str, Any]]:
        """
        Convert predicted bounding box and ID into a dictionary organized by frame number.

        Returns
        -------
        Dict[int, Dict[str, Any]]:
            A dictionary where the key is the frame number and the value is another dictionary containing:
            - 'bbox': A numpy array with shape (N, 4) containing coordinates of the bounding boxes
            [x, y, x + width, y + height] for every object in the frame.
            - 'id': A numpy array containing the IDs of the tracked objects.
        """
        predicted_dict: Dict[int, Dict[str, Any]] = {}

        for frame_number, frame_data in enumerate(self.predicted_boxes_id):
            if frame_data.size == 0:
                continue

            bboxes = frame_data[:, :4]
            ids = frame_data[:, 4]

            predicted_dict[frame_number] = {"bbox": bboxes, "id": ids}

        return predicted_dict

    def get_ground_truth_data(self) -> Dict[int, Dict[str, Any]]:
        """
        Extract ground truth bounding box data from a CSV file and organize it by frame number.

        Returns
        -------
        Dict[int, Dict[str, Any]]:
            A dictionary where the key is the frame number and the value is another dictionary containing:
            - 'bbox': A numpy arrays with shape of (N, 4) containing coordinates of the bounding box
                [x, y, x + width, y + height] for every crabs in the frame.
            - 'id': The ground truth ID
        """
        with open(self.gt_dir, "r") as csvfile:
            csvreader = csv.reader(csvfile)
            next(csvreader)  # Skip the header row
            ground_truth_data = [
                extract_bounding_box_info(row) for row in csvreader
            ]

        # Format as a dictionary with key = frame number
        ground_truth_dict: dict = {}
        for data in ground_truth_data:
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

            if frame_number not in ground_truth_dict:
                ground_truth_dict[frame_number] = {"bbox": [], "id": []}

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
        """
        Calculate IoU (Intersection over Union) of two bounding boxes.

        Parameters
        ----------
        box1 (np.ndarray):
            Coordinates [x1, y1, x2, y2] of the first bounding box.
            Here, (x1, y1) represents the top-left corner, and (x2, y2) represents the bottom-right corner.
        box2 (np.ndarray):
            Coordinates [x1, y1, x2, y2] of the second bounding box.
            Here, (x1, y1) represents the top-left corner, and (x2, y2) represents the bottom-right corner.

        Returns
        -------
        float:
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
                    self.last_known_ids[gt_id] = pred_id
            return 0

        switch_counter = 0
        # Compute sets of ground truth IDs for current and previous frames
        gt_ids_current_frame = set(gt_to_tracked_id_current_frame.keys())
        gt_ids_prev_frame = set(gt_to_tracked_id_previous_frame.keys())

        # Compute lists of ground truth IDs that continue, disappear, and appear
        gt_ids_cont = list(gt_ids_current_frame & gt_ids_prev_frame)
        gt_ids_disappear = list(gt_ids_prev_frame - gt_ids_current_frame)
        gt_ids_appear = list(gt_ids_current_frame - gt_ids_prev_frame)

        # Store used predicted IDs to avoid double counting
        # In `used_pred_ids` we log IDs from either the current or the previous frame that have been involved in an already counted ID switch.
        used_pred_ids = set()

        # Case 1: Objects that continue to exist
        for gt_id in gt_ids_cont:
            previous_pred_id = gt_to_tracked_id_previous_frame.get(gt_id)
            current_pred_id = gt_to_tracked_id_current_frame.get(gt_id)
            if not np.isnan(previous_pred_id) and not np.isnan(
                current_pred_id
            ):
                if current_pred_id != previous_pred_id:
                    switch_counter += 1
                    used_pred_ids.add(current_pred_id)
		# save most recent predicted ID associated to this groundtruth ID
                self.last_known_ids[gt_id] = current_pred_id

        # Case 2: Objects that disappear
        for gt_id in gt_ids_disappear:
            previous_pred_id = gt_to_tracked_id_previous_frame.get(gt_id)
            if not np.isnan(
                previous_pred_id
            ):  # Exclude if missed detection in previous frame
                if previous_pred_id in gt_to_tracked_id_current_frame.values():
                    if previous_pred_id not in used_pred_ids:
                        switch_counter += 1
                        used_pred_ids.add(previous_pred_id)

        # Case 3: Objects that appear
        for gt_id in gt_ids_appear:
            current_pred_id = gt_to_tracked_id_current_frame.get(gt_id)
            if not np.isnan(current_pred_id):
                if current_pred_id in gt_to_tracked_id_previous_frame.values():
                    if previous_pred_id not in used_pred_ids:
                        switch_counter += 1
                elif gt_id in self.last_known_ids.keys():
                    last_known_predicted_id = self.last_known_ids[gt_id]
                    if current_pred_id != last_known_id:
                        switch_counter += 1
                self.last_known_ids[gt_id] = current_pred_id

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
                    iou = self.calculate_iou(gt_box, pred_box)
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
        self.total_num_switches += num_switches
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

    def run_evaluation(self) -> None:
        """
        Run evaluation of tracking based on tracking ground truth.
        """
        predicted_dict = self.get_predicted_data()
        ground_truth_dict = self.get_ground_truth_data()
        mota_values = self.evaluate_tracking(ground_truth_dict, predicted_dict)
        print(self.total_num_switches)
        overall_mota = np.mean(mota_values)
        logging.info("Overall MOTA: %f" % overall_mota)
