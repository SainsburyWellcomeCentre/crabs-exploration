import csv
import datetime
import json
import logging
import os
from pathlib import Path
from typing import Any, Dict, List, Optional

import cv2
import numpy as np
import torch


def coco_category():
    """
    Get the COCO instance category names.

    Returns
    -------
    list of str
        List of COCO instance category names.
    """
    COCO_INSTANCE_CATEGORY_NAMES = [
        "__background__",
        "crab",
    ]
    return COCO_INSTANCE_CATEGORY_NAMES


def save_model(model: torch.nn.Module):
    """
    Save the trained model.

    Parameters
    ----------
    model : torch.nn.Module
        The PyTorch model to be saved.

    Returns
    -------
    None

    Notes
    -----
    This function saves the provided PyTorch model to a file with a unique
    filename based on the current date and time. The filename format is
    'model_<timestamp>.pt'.

    """
    current_time = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    directory = "model"
    os.makedirs(directory, exist_ok=True)
    filename = f"{directory}/model_{current_time}.pt"

    print(filename)
    torch.save(model, filename)
    print("Model Saved")


def draw_bbox(
    frame: np.ndarray,
    top_pt: int,
    left_pt: int,
    bottom_pt: int,
    right_pt: int,
    colour: tuple,
    label_text: Optional[str] = None,
) -> None:
    """
    Draw bounding boxes on the image based on detection results.

    Parameters
    ----------
    frame : np.ndarray
        Image with bounding boxes drawn on it.
    top_pt : int
        Y-coordinate of the top-left corner of the bounding box.
    left_pt : int
        X-coordinate of the top-left corner of the bounding box.
    bottom_pt : int
        Y-coordinate of the bottom-right corner of the bounding box.
    right_pt : int
        X-coordinate of the bottom-right corner of the bounding box.
    colour : tuple
        Color of the bounding box in BGR format.
    label_text : str, optional
        Text to display alongside the bounding box, indicating class and score.

    Returns
    -------
    None
    """
    # Draw bounding box
    cv2.rectangle(
        frame,
        (top_pt, left_pt),
        (bottom_pt, right_pt),
        colour,
        thickness=2,
    )

    # Add label text if provided
    if label_text:
        cv2.putText(
            frame,
            label_text,
            (top_pt, left_pt),
            cv2.FONT_HERSHEY_SIMPLEX,
            1,
            colour,
            2,
            cv2.LINE_AA,
        )


def draw_detection(
    imgs: list,
    annotations: dict,
    detections: Optional[Dict[Any, Any]] = None,
    score_threshold: Optional[float] = None,
) -> np.ndarray:
    """
    Draw the results based on the detection.

    Parameters
    ----------
    imgs : list
        List of images.
    annotations : dict
        Ground truth annotations.
    detections : dict, optional
        Detected objects.
    score_threshold : float, optional
        The confidence threshold for detection scores.

    Returns
    -------
    np.ndarray
        Image(s) with bounding boxes drawn on them.
    """
    coco_list = coco_category()
    image_with_boxes = None

    for image, label, prediction in zip(
        imgs, annotations, detections or [None] * len(imgs)
    ):
        image = image.cpu().numpy().transpose(1, 2, 0)
        image = (image * 255).astype("uint8")
        image_with_boxes = image.copy()

        target_boxes = [
            [(i[0], i[1]), (i[2], i[3])]
            for i in list(label["boxes"].detach().cpu().numpy())
        ]

        for i in range(len(target_boxes)):
            draw_bbox(
                image_with_boxes,
                int((target_boxes[i][0])[0]),
                int((target_boxes[i][0])[1]),
                int((target_boxes[i][1])[0]),
                int((target_boxes[i][1])[1]),
                colour=(0, 255, 0),
            )
        if prediction:
            pred_score = list(prediction["scores"].detach().cpu().numpy())
            pred_t = [pred_score.index(x) for x in pred_score][-1]

            pred_class = [
                coco_list[i]
                for i in list(prediction["labels"].detach().cpu().numpy())
            ]

            pred_boxes = [
                [(i[0], i[1]), (i[2], i[3])]
                for i in list(
                    prediction["boxes"].detach().cpu().detach().numpy()
                )
            ]

            pred_boxes = pred_boxes[: pred_t + 1]
            pred_class = pred_class[: pred_t + 1]
            for i in range(len(pred_boxes)):
                if pred_score[i] > (score_threshold or 0):
                    label_text = f"{pred_class[i]}: {pred_score[i]:.2f}"
                    draw_bbox(
                        image_with_boxes,
                        int((pred_boxes[i][0])[0]),
                        int((pred_boxes[i][0])[1]),
                        int((pred_boxes[i][1])[0]),
                        int((pred_boxes[i][1])[1]),
                        (0, 0, 255),
                        label_text,
                    )
    return image_with_boxes


def calculate_iou(box1: np.ndarray, box2: np.ndarray) -> float:
    """
    Calculate IoU (Intersection over Union) of two bounding boxes.

    Parameters:
    box1 (np.ndarray): Coordinates [x1, y1, x2, y2] of the first bounding box.
    box2 (np.ndarray): Coordinates [x1, y1, x2, y2] of the second bounding box.

    Returns:
    float: IoU value.
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
    prev_frame_ids: Optional[List[List[int]]],
    current_frame_ids: Optional[List[List[int]]],
) -> int:
    """
    Count the number of identity switches between two sets of object IDs.

    Parameters:
    -----------
    prev_frame_ids : Optional[List[List[int]]]
        List of object IDs in the previous frame.
    current_frame_ids : Optional[List[List[int]]]
        List of object IDs in the current frame.

    Returns:
    --------
    int
        The number of identity switches between the two sets of object IDs.
    """

    if prev_frame_ids is None or current_frame_ids is None:
        return 0

    # Initialize count of identity switches
    num_switches = 0

    prev_ids = set(prev_frame_ids[0]) if prev_frame_ids else set()
    current_ids = set(current_frame_ids[0])

    # Calculate the number of switches by finding the difference in IDs
    num_switches = len(current_ids - prev_ids) - len(prev_ids - current_ids)

    return num_switches


def evaluate_mota(
    gt_boxes: np.ndarray,
    tracked_boxes: np.ndarray,
    iou_threshold: float,
    prev_frame_ids: Optional[List[List[int]]],
) -> float:
    """
    Evaluate MOTA (Multiple Object Tracking Accuracy).

    MOTA is a metric used to evaluate the performance of object tracking algorithms.

    Parameters:
    -----------
    gt_boxes : np.ndarray
        Ground truth bounding boxes of objects.
    tracked_boxes : np.ndarray
        Tracked bounding boxes of objects.
    iou_threshold : float
        Intersection over Union (IoU) threshold for considering a match.
    prev_frame_ids : Optional[List[List[int]]]
        IDs from the previous frame for identity switch detection.

    Returns:
    --------
    float
        The computed MOTA (Multi-Object Tracking Accuracy) score for the tracking performance.
    """
    total_gt = len(gt_boxes)
    false_alarms = 0

    for i, tracked_box in enumerate(tracked_boxes):
        best_iou = 0.0
        best_match = None

        for j, gt_box in enumerate(gt_boxes):
            iou = calculate_iou(gt_box[:4], tracked_box[:4])
            if iou > iou_threshold and iou > best_iou:
                best_iou = iou
                best_match = j
        if best_match is not None:
            gt_boxes[best_match] = None
        else:
            false_alarms += 1

    missed_detections = 0
    for box in gt_boxes:
        if box is not None and not np.all(np.isnan(box)):
            missed_detections += 1

    tracked_ids = [[int(box[-1]) for box in tracked_boxes]]

    num_switches = count_identity_switches(prev_frame_ids, tracked_ids)
    mota = 1 - (missed_detections + false_alarms + num_switches) / total_gt
    return mota


def extract_bounding_box_info(row: List[str]) -> Dict[str, Any]:
    """
    Extracts bounding box information from a row of data.

    Parameters:
    -----------
    row : List[str]
        A list representing a row of data containing information about a bounding box.

    Returns:
    --------
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
    ground_truth_data: List[Dict[str, Any]], gt_boxes_list: List[np.ndarray]
) -> List[np.ndarray]:
    """
    Creates a list of ground truth bounding boxes organized by frame number.

    Parameters:
    -----------
    ground_truth_data : List[Dict[str, Any]]
        A list containing ground truth bounding box data organized by frame number.
    gt_boxes_list : List[np.ndarray]
        A list to store the ground truth bounding boxes for each frame.

    Returns:
    --------
    List[np.ndarray]:
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


def get_ground_truth_data(gt_dir: str) -> List[np.ndarray]:
    """
    Extract ground truth bounding box data from a CSV file.

    Parameters:
    -----------
    gt_dir : str
        The path to the CSV file containing ground truth data.

    Returns:
    --------
    List[np.ndarray]:
        A list containing ground truth bounding box data organized by frame number.
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


def write_bbox_to_csv(
    bbox: np.ndarray,
    frame: np.ndarray,
    frame_name: str,
    csv_writer: Any,
) -> None:
    """
    Write bounding box annotation to a CSV file.

    Parameters:
    -----------
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
            '{{"track":"{}"}}'.format(id),
        )
    )


def save_frame_and_csv(
    video_file_root: str,
    tracking_output_dir: Path,
    tracked_boxes: List[List[float]],
    frame: np.ndarray,
    frame_number: int,
    csv_writer: Any,
) -> None:
    """
    Common functionality for saving frames and CSV
    """
    for bbox in tracked_boxes:
        # Get frame name
        frame_name = f"{video_file_root}frame_{frame_number:08d}.png"

        # Add bbox to csv
        write_bbox_to_csv(bbox, frame, frame_name, csv_writer)

        # Save frame as PNG
        frame_path = tracking_output_dir / frame_name
        img_saved = cv2.imwrite(str(frame_path), frame)
        if img_saved:
            logging.info(f"Frame {frame_number} saved at {frame_path}")
        else:
            logging.info(
                f"ERROR saving {frame_name}, frame {frame_number}...skipping"
            )
            break
