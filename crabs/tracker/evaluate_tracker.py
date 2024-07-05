import csv
import logging
from typing import Any, Dict, Optional, Tuple

import numpy as np

from crabs.tracker.utils.tracking import extract_bounding_box_info


class TrackerEvaluate:
    def __init__(
        self, gt_dir: str, tracked_list: list[np.ndarray], iou_threshold: float
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
        self.tracked_list = tracked_list
        self.iou_threshold = iou_threshold

    def get_ground_truth_data(self) -> Dict[int, Dict[str, Any]]:
        """
        Extract ground truth bounding box data from a CSV file and organize it by frame number.

        Returns
        -------
        Dict[int, Dict[str, Any]]:
            A dictionary where the key is the frame number and the value is another dictionary containing:
            - 'bbox': A list of numpy arrays with coordinates of the bounding box [x, y, x + width, y + height]
            - 'id': The ground truth ID
        """
        ground_truth_data = []

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
        prev_frame_id_map: Optional[Dict[int, int]],
        current_frame_id_map: Dict[int, int],
    ) -> int:
        """
        Count the number of identity switches between two sets of object IDs.

        Parameters
        ----------
        prev_frame_id_map : Optional[Dict[int, int]]
            A dictionary mapping ground truth IDs to predicted IDs from the previous frame.
        gt_to_tracked_map : Dict[int, int]
            A dictionary mapping ground truth IDs to predicted IDs for the current frame.


        Returns
        -------
        int
            The number of identity switches between the two sets of object IDs.
        """

        if prev_frame_id_map is None:
            return 0

        prev_frame_gt_id_map = {v: k for k, v in prev_frame_id_map.items()}

        switch_count = 0

        for (
            current_gt_id,
            current_predicted_id,
        ) in current_frame_id_map.items():
            if np.isnan(current_predicted_id):
                continue
            prev_tracked_id = prev_frame_id_map.get(current_gt_id)
            prev_gt_id = prev_frame_gt_id_map.get(current_predicted_id)
            if prev_tracked_id is not None:
                if prev_tracked_id != current_predicted_id:
                    switch_count += 1
            elif prev_gt_id is not None:
                if current_gt_id != prev_gt_id:
                    switch_count += 1

        return switch_count

    def evaluate_mota(
        self,
        gt_boxes: np.ndarray,
        gt_ids: np.ndarray,
        tracked_boxes: np.ndarray,
        iou_threshold: float,
        prev_frame_id_map: Optional[Dict[int, int]],
    ) -> Tuple[float, Dict[int, int]]:
        """
        Evaluate MOTA (Multiple Object Tracking Accuracy).

        MOTA is a metric used to evaluate the performance of object tracking algorithms.

        Parameters
        ----------
        gt_boxes : np.ndarray
            Ground truth bounding boxes of objects with shape of (N, 4) with (x1, y1, x2, y2).
        gt_ids : np.ndarray
            Ground truth IDs corresponding to the bounding boxes with shape of (N, 1).
        tracked_boxes : np.ndarray
            Tracked bounding boxes of objects with shape of (N, 5) with (x1, y1, x2, y2, id).
        iou_threshold : float
            Intersection over Union (IoU) threshold for considering a match.
        prev_frame_id_map : Optional[Dict[int, int]]
            A dictionary mapping ground truth IDs to predicted IDs from the previous frame.

        Returns
        -------
        float
            The computed MOTA (Multi-Object Tracking Accuracy) score for the tracking performance.
        Dict[int, int]
            A dictionary mapping ground truth IDs to predicted IDs for the current frame.

        Notes
        -----
        MOTA is calculated using the following formula:

        MOTA = 1 - (Missed Detections + False Positives + Identity Switches) / Total Ground Truth

        - Missed Detections: Instances where the ground truth objects were not detected by the tracking algorithm.
        - False Positives: Instances where the tracking algorithm produces a detection where there is no corresponding ground truth object.
        - Identity Switches: Instances where the tracking algorithm assigns a different ID to an object compared to its ID in the previous frame.
        - Total Ground Truth: The total number of ground truth objects in the scene.

        The MOTA score ranges from -inf to 1, with higher values indicating better tracking performance.
        A MOTA score of 1 indicates perfect tracking, where there are no missed detections, false positives, or identity switches.
        """
        total_gt = len(gt_boxes)
        false_positive = 0
        matched_gt_boxes = set()
        gt_to_tracked_map = {}

        for i, tracked_box in enumerate(tracked_boxes):
            best_iou = 0.0
            best_match = None
            miss_track_id = None

            for j, gt_box in enumerate(gt_boxes):
                if j not in matched_gt_boxes:
                    iou = self.calculate_iou(gt_box[:4], tracked_box[:4])
                    # print(iou)
                    if iou > iou_threshold and iou > best_iou:
                        best_iou = iou
                        best_match = j
                    else:
                        miss_track_id = j
                        print(miss_track_id)

            if best_match is not None:
                # successfully found a matching ground truth box for the tracked box.
                matched_gt_boxes.add(best_match)
                # Map ground truth ID to tracked ID
                gt_to_tracked_map[int(gt_ids[best_match])] = int(
                    tracked_box[-1]
                )
            else:
                false_positive += 1
            if miss_track_id is not None:
                gt_to_tracked_map[int(gt_ids[miss_track_id])] = np.nan

        missed_detections = total_gt - len(matched_gt_boxes)
        num_switches = self.count_identity_switches(
            prev_frame_id_map, gt_to_tracked_map
        )

        mota = (
            1 - (missed_detections + false_positive + num_switches) / total_gt
        )
        return mota, gt_to_tracked_map

    def evaluate_tracking(
        self,
        ground_truth_dict: Dict[int, Dict[str, Any]],
    ) -> list[float]:
        """
        Evaluate tracking performance using the Multi-Object Tracking Accuracy (MOTA) metric.

        Parameters
        ----------
        ground_truth_dict : dict
            Dictionary containing ground truth bounding boxes and IDs for each frame, organized by frame number.

        Returns
        -------
        list[float]:
            The computed MOTA (Multi-Object Tracking Accuracy) score for the tracking performance.
        """
        mota_values = []
        prev_frame_id_map: Optional[dict] = None

        for frame_number in sorted(ground_truth_dict.keys()):
            gt_data = ground_truth_dict[frame_number]
            gt_boxes = np.array(
                [[x1, y1, x2, y2] for x1, y1, x2, y2 in gt_data["bbox"]],
                dtype=np.float32,
            )
            gt_ids = np.array(gt_data["id"], dtype=np.float32)

            if frame_number < len(self.tracked_list):
                tracked_boxes = self.tracked_list[frame_number]
                mota, prev_frame_id_map = self.evaluate_mota(
                    gt_boxes,
                    gt_ids,
                    tracked_boxes,
                    self.iou_threshold,
                    prev_frame_id_map,
                )
                mota_values.append(mota)

        return mota_values

    def run_evaluation(self) -> None:
        """
        Run evaluation of tracking based on tracking ground truth.
        """
        ground_truth_dict = self.get_ground_truth_data()
        mota_values = self.evaluate_tracking(ground_truth_dict)
        overall_mota = np.mean(mota_values)
        logging.info("Overall MOTA: %f" % overall_mota)
