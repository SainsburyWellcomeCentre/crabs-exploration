import csv
import logging
from typing import Any, Dict, Optional

import numpy as np

from crabs.tracker.utils.tracking import extract_bounding_box_info


class TrackerEvaluate:
    def __init__(self, gt_dir: str, tracked_list: list, iou_threshold: float):
        self.gt_dir = gt_dir
        self.tracked_list = tracked_list
        self.iou_threshold = iou_threshold

    def create_gt_list(
        self,
        ground_truth_data: list[Dict[str, Any]],
        gt_boxes_list: list[np.ndarray],
    ) -> list[np.ndarray]:
        """
        Creates a list of ground truth bounding boxes organized by frame number.

        Parameters
        ----------
        ground_truth_data : list[Dict[str, Any]]
            A list containing ground truth bounding box data organized by frame number.
        gt_boxes_list : list[np.ndarray]
            A list to store the ground truth bounding boxes for each frame.

        Returns
        -------
        list[np.ndarray]:
            A list containing ground truth bounding boxes organized by frame number.
        """
        for data in ground_truth_data:
            frame_number = data["frame_number"]
            bbox = np.array(
                [
                    data["x"],
                    data["y"],
                    data["x"] + data["width"],
                    data["y"] + data["height"],
                    data["id"],
                ],
                dtype=np.float32,
            )
            if gt_boxes_list[frame_number].size == 0:
                gt_boxes_list[frame_number] = bbox.reshape(
                    1, -1
                )  # Initialize as a 2D array
            else:
                gt_boxes_list[frame_number] = np.vstack(
                    [gt_boxes_list[frame_number], bbox]
                )
        return gt_boxes_list

    def get_ground_truth_data(self) -> list[np.ndarray]:
        """
        Extract ground truth bounding box data from a CSV file.

        Parameters
        ----------
        gt_dir : str
            The path to the CSV file containing ground truth data.

        Returns
        -------
        list[np.ndarray]:
            A list containing ground truth bounding box data organized by frame number.
            The numpy array represent the coordinates and ID of the bounding box in the order:
            x, y, x + width, y + height, ID
        """
        ground_truth_data = []
        max_frame_number = 0

        # Open the CSV file and read its contents line by line
        with open(self.gt_dir, "r") as csvfile:
            csvreader = csv.reader(csvfile)
            next(csvreader)  # Skip the header row
            for row in csvreader:
                data = extract_bounding_box_info(row)
                ground_truth_data.append(data)
                max_frame_number = max(max_frame_number, data["frame_number"])

        # Initialize a list to store the ground truth bounding boxes for each frame
        gt_boxes_list = [np.array([]) for _ in range(max_frame_number + 1)]

        gt_boxes_list = self.create_gt_list(ground_truth_data, gt_boxes_list)
        return gt_boxes_list

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
        prev_frame_ids: Optional[list[list[int]]],
        current_frame_ids: Optional[list[list[int]]],
    ) -> int:
        """
        Count the number of identity switches between two sets of object IDs.

        Parameters
        ----------
        prev_frame_ids : Optional[list[list[int]]]
            List of object IDs in the previous frame.
        current_frame_ids : Optional[list[list[int]]]
            List of object IDs in the current frame.

        Returns
        -------
        int
            The number of identity switches between the two sets of object IDs.
        """

        if prev_frame_ids is None or current_frame_ids is None:
            return 0

        # Initialize count of identity switches
        num_switches = 0

        prev_ids = set(prev_frame_ids[0])
        current_ids = set(current_frame_ids[0])

        # Calculate the number of switches by finding the difference in IDs
        num_switches = len(prev_ids.symmetric_difference(current_ids))

        return num_switches

    def evaluate_mota(
        self,
        gt_boxes: np.ndarray,
        tracked_boxes: np.ndarray,
        iou_threshold: float,
        prev_frame_ids: Optional[list[list[int]]],
    ) -> float:
        """
        Evaluate MOTA (Multiple Object Tracking Accuracy).

        MOTA is a metric used to evaluate the performance of object tracking algorithms.

        Parameters
        ----------
        gt_boxes : np.ndarray
            Ground truth bounding boxes of objects.
        tracked_boxes : np.ndarray
            Tracked bounding boxes of objects.
        iou_threshold : float
            Intersection over Union (IoU) threshold for considering a match.
        prev_frame_ids : Optional[list[list[int]]]
            IDs from the previous frame for identity switch detection.

        Returns
        -------
        float
            The computed MOTA (Multi-Object Tracking Accuracy) score for the tracking performance.

        Notes
        -----
        MOTA is calculated using the following formula:

        MOTA = 1 - (Missed Detections + False Positives + Identity Switches) / Total Ground Truth

        - Missed Detections: Instances where the ground truth objects were not detected by the tracking algorithm.
        - False Positives: Instances where the tracking algorithm produces a detection where there is no corresponding ground truth object.
        - Identity Switches: Instances where the tracking algorithm assigns a different ID to an object compared to its ID in the previous frame.
        - Total Ground Truth: The total number of ground truth objects in the scene.

        The MOTA score ranges from 0 to 1, with higher values indicating better tracking performance.
        A MOTA score of 1 indicates perfect tracking, where there are no missed detections, false positives, or identity switches.
        """
        total_gt = len(gt_boxes)
        false_positive = 0

        for i, tracked_box in enumerate(tracked_boxes):
            best_iou = 0.0
            best_match = None

            for j, gt_box in enumerate(gt_boxes):
                iou = self.calculate_iou(gt_box[:4], tracked_box[:4])
                if iou > iou_threshold and iou > best_iou:
                    best_iou = iou
                    best_match = j
            if best_match is not None:
                # successfully found a matching ground truth box for the tracked box.
                # set the corresponding ground truth box to None.
                gt_boxes[best_match] = None
            else:
                false_positive += 1

        missed_detections = 0
        for box in gt_boxes:
            if box is not None and not np.all(np.isnan(box)):
                # if true ground truth box was not matched with any tracked box
                missed_detections += 1

        tracked_ids = [[box[-1] for box in tracked_boxes]]

        num_switches = self.count_identity_switches(
            prev_frame_ids, tracked_ids
        )

        mota = (
            1 - (missed_detections + false_positive + num_switches) / total_gt
        )
        return mota

    def evaluate_tracking(self, gt_boxes_list: list) -> list[float]:
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
        for gt_boxes, tracked_boxes in zip(gt_boxes_list, self.tracked_list):
            mota = self.evaluate_mota(
                gt_boxes,
                tracked_boxes,
                self.iou_threshold,
                prev_frame_ids,
            )
            mota_values.append(mota)
            # Update previous frame IDs for the next iteration
            prev_frame_ids = [[box[-1] for box in tracked_boxes]]

        return mota_values

    def run_evaluation(self):
        gt_boxes_list = self.get_ground_truth_data()
        mota_values = self.evaluate_tracking(gt_boxes_list)
        overall_mota = np.mean(mota_values)
        logging.info("Overall MOTA:", overall_mota)
