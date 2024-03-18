import csv
import datetime
import json
import logging
import os
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

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
    annotations: Optional[Dict[Any, Any]] = None,
    detections: Optional[Dict[Any, Any]] = None,
    score_threshold: Optional[float] = None,
) -> np.ndarray:
    """
    Draw the results based on the detection.

    Parameters
    ----------
    imgs : list
        List of images.
    annotations : dict, optional
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
        imgs, annotations or [], detections or []
    ):
        image = image.cpu().numpy().transpose(1, 2, 0)
        image = (image * 255).astype("uint8")
        image_with_boxes = image.copy()

        if label:
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


def draw_gt_tracking(
    gt_boxes: np.ndarray,
    tracked_boxes: np.ndarray,
    frame_number: int,
    iou_threshold: float,
    frame_copy: np.ndarray,
) -> np.ndarray:
    """
    Track ground truth objects in the frame and draw bounding boxes.

    Parameters
    ----------
    gt_boxes : np.ndarray
        An array containing ground truth bounding boxes of objects for the current frame.
    tracked_boxes : np.ndarray
        An array containing sorted bounding boxes of detected objects.
    frame_number : int
        The frame number to track.
    iou_threshold : float
        The intersection over union threshold for considering a match.
    frame_copy : np.ndarray
        A copy of the input frame for drawing bounding boxes.

    Returns
    -------
    np.ndarray
        A copy of the input frame with bounding boxes drawn on it.
    """

    for gt_box in gt_boxes:
        x_gt, y_gt, x2_gt, y2_gt, gt_id = gt_box

        for tracked_box in tracked_boxes:
            x1_track, y1_track, x2_track, y2_track, track_id = tracked_box
            iou = calculate_iou(
                [x_gt, y_gt, x2_gt, y2_gt],
                [x1_track, y1_track, x2_track, y2_track],
            )
            x_gt, y_gt, x2_gt, y2_gt = map(int, [x_gt, y_gt, x2_gt, y2_gt])
            x1_track, y1_track, x2_track, y2_track = map(
                int, [x1_track, y1_track, x2_track, y2_track]
            )

            if iou > iou_threshold:
                draw_bbox(
                    frame_copy,
                    x_gt,
                    y_gt,
                    x2_gt,
                    y2_gt,
                    (0, 255, 0),
                    f"gt id : {int(gt_id)}",
                )

                draw_bbox(
                    frame_copy,
                    x1_track,
                    y1_track,
                    x2_track,
                    y2_track,
                    (0, 0, 255),
                    f"track id : {int(track_id)}",
                )

    return frame_copy


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
    prev_frame: Optional[List[List[float]]], current_frame: List[List[float]]
) -> int:
    """
    Count the number of identity switches between two sets of object IDs.
    """
    if prev_frame is None:
        # If there are no previous frame IDs, return 0 switches
        return 0

    # Convert tracked boxes to tuples for comparison
    prev_frame_tuples = [tuple(box) for box in prev_frame]
    current_frame_tuples = [tuple(box) for box in current_frame]

    # Create dictionaries to track object IDs in each frame
    id_to_index_prev = {id_: i for i, id_ in enumerate(prev_frame_tuples)}
    id_to_index_current = {
        id_: i for i, id_ in enumerate(current_frame_tuples)
    }

    # Initialize count of identity switches
    num_switches = 0

    # Loop through object IDs in the current frame
    for id_current, index_current in id_to_index_current.items():
        # Check if the object ID exists in the previous frame
        if id_current in id_to_index_prev:
            # Get the corresponding index in the previous frame
            index_prev = id_to_index_prev[id_current]
            # If the index is different, it indicates an identity switch
            if index_current != index_prev:
                num_switches += 1

    return num_switches


def evaluate_mota(
    gt_boxes: np.ndarray,
    tracked_boxes: np.ndarray,
    iou_threshold: float,
    prev_frame: Union[Optional[List[List[float]]], None],
) -> float:
    """
    Evaluate MOTA (Multiple Object Tracking Accuracy).

    MOTA is a metric used to evaluate the performance of object tracking algorithms.
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

    num_switches = count_identity_switches(prev_frame, tracked_boxes)
    mota = 1 - (missed_detections + false_alarms + num_switches) / total_gt
    return mota


def get_ground_truth_data(gt_dir: str) -> list:
    """
    Extract ground truth bounding box data from a CSV file.

    Parameters:
    -----------
    gt_dir : str
        The path to the CSV file containing ground truth data.

    Returns:
    --------
    list:
        A list containing ground truth bounding box data organized by frame number.
    """
    ground_truth_data = []
    max_frame_number = 0

    # Open the CSV file and read its contents line by line
    with open(gt_dir, "r") as csvfile:
        csvreader = csv.reader(csvfile)
        next(csvreader)  # Skip the header row
        for row in csvreader:
            # Extract relevant information from each row
            filename = row[0]
            region_shape_attributes = json.loads(row[5])
            region_attributes = json.loads(row[6])

            # Extract bounding box coordinates and object ID
            x = region_shape_attributes["x"]
            y = region_shape_attributes["y"]
            width = region_shape_attributes["width"]
            height = region_shape_attributes["height"]
            track_id = region_attributes["track"]

            # Compute the frame number from the filename
            frame_number = int(filename.split("_")[-1].split(".")[0])
            frame_number = frame_number - 1

            # Update max_frame_number
            max_frame_number = max(max_frame_number, frame_number)

            # Append the extracted data to the list
            ground_truth_data.append(
                {
                    "frame_number": frame_number,
                    "x": x,
                    "y": y,
                    "width": width,
                    "height": height,
                    "id": track_id,
                }
            )

    # Initialize a list to store the ground truth bounding boxes for each frame
    gt_boxes_list = [np.array([]) for _ in range(max_frame_number + 1)]

    # Organize ground truth data into gt_boxes_list
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
        # print(bbox)
        gt_boxes_list[frame_number] = (
            np.vstack([gt_boxes_list[frame_number], bbox])
            if gt_boxes_list[frame_number].size
            else bbox
        )
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
    save_plot: bool = True,
) -> np.ndarray:
    """
    Common functionality for saving frames and CSV
    """
    frame_copy = frame.copy()

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

        if save_plot:
            # Plot
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

    return frame_copy
