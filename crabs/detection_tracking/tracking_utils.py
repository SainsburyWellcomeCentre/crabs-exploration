import csv
import json
import logging
from pathlib import Path
from typing import Any, Dict, Optional

import cv2
import numpy as np


def calculate_iou(box1: np.ndarray, box2: np.ndarray) -> float:
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
            iou = calculate_iou(gt_box[:4], tracked_box[:4])
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

    num_switches = count_identity_switches(prev_frame_ids, tracked_ids)
    print(total_gt)
    print(missed_detections)
    print(false_positive)
    print(num_switches)

    mota = 1 - (missed_detections + false_positive + num_switches) / total_gt
    return mota


def extract_bounding_box_info(row: list[str]) -> Dict[str, Any]:
    """
    Extracts bounding box information from a row of data.

    Parameters
    ----------
    row : list[str]
        A list representing a row of data containing information about a bounding box.

    Returns
    -------
    Dict[str, Any]:
        A dictionary containing the extracted bounding box information.
    """
    filename = row[0]
    region_shape_attributes = json.loads(row[5])
    region_attributes = json.loads(row[6])

    x = region_shape_attributes["x"]
    y = region_shape_attributes["y"]
    width = region_shape_attributes["width"]
    height = region_shape_attributes["height"]
    track_id = region_attributes["track"]

    frame_number = int(filename.split("_")[-1].split(".")[0]) - 1
    return {
        "frame_number": frame_number,
        "x": x,
        "y": y,
        "width": width,
        "height": height,
        "id": track_id,
    }


def create_gt_list(
    ground_truth_data: list[Dict[str, Any]], gt_boxes_list: list[np.ndarray]
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


def get_ground_truth_data(gt_dir: str) -> list[np.ndarray]:
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
    with open(gt_dir, "r") as csvfile:
        csvreader = csv.reader(csvfile)
        next(csvreader)  # Skip the header row
        for row in csvreader:
            data = extract_bounding_box_info(row)
            ground_truth_data.append(data)
            max_frame_number = max(max_frame_number, data["frame_number"])

    # Initialize a list to store the ground truth bounding boxes for each frame
    gt_boxes_list = [np.array([]) for _ in range(max_frame_number + 1)]

    gt_boxes_list = create_gt_list(ground_truth_data, gt_boxes_list)
    return gt_boxes_list


def write_tracked_bbox_to_csv(
    bbox: np.ndarray,
    frame: np.ndarray,
    frame_name: str,
    csv_writer: Any,
) -> None:
    """
    Write bounding box annotation to a CSV file.

    Parameters
    ----------
    bbox : np.ndarray
        A numpy array containing the bounding box coordinates
        (xmin, ymin, xmax, ymax, id).
    frame : np.ndarray
        The frame to which the bounding box belongs.
    frame_name : str
        The name of the frame.
    csv_writer : Any
        The CSV writer object to write the annotation.
    """
    # Bounding box geometry
    xmin, ymin, xmax, ymax, id = bbox
    width_box = int(xmax - xmin)
    height_box = int(ymax - ymin)

    # Add to csv
    csv_writer.writerow(
        (
            frame_name,
            frame.size,
            '{{"clip":{}}}'.format("123"),
            1,
            0,
            '{{"name":"rect","x":{},"y":{},"width":{},"height":{}}}'.format(
                xmin, ymin, width_box, height_box
            ),
            '{{"track":"{}"}}'.format(int(id)),
        )
    )


def save_frame_and_csv(
    video_file_root: str,
    tracking_output_dir: Path,
    tracked_boxes: list[list[float]],
    frame: np.ndarray,
    frame_number: int,
    csv_writer: Any,
) -> None:
    """
    Save tracked bounding boxes as frames and write to a CSV file.

    Parameters
    ----------
    video_file_root : str
        The root path of the video file.
    tracking_output_dir : Path
        The directory where tracked frames and CSV file will be saved.
    tracked_boxes : list[list[float]]
        List of bounding boxes to be saved.
    frame : np.ndarray
        The frame image.
    frame_number : int
        The frame number.
    csv_writer : Any
        CSV writer object for writing bounding box data.

    Returns
    -------
    None
    """
    frame_name = f"{video_file_root}_frame_{frame_number:08d}.png"

    for bbox in tracked_boxes:
        # Add bbox to csv
        write_tracked_bbox_to_csv(bbox, frame, frame_name, csv_writer)

    # Save frame as PNG - once as per frame
    frame_path = tracking_output_dir / frame_name
    img_saved = cv2.imwrite(str(frame_path), frame)
    if not img_saved:
        logging.error(
            f"Didn't save {frame_name}, frame {frame_number}, Skipping."
        )
    logging.info(f"Frame {frame_number} saved at {frame_path}")
