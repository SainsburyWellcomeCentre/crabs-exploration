import csv
import logging
from typing import Any, Dict, Optional

import numpy as np

from crabs.tracking._utils import evaluate_mota, extract_bounding_box_info


class Evaluation:
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
            mota = evaluate_mota(
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
