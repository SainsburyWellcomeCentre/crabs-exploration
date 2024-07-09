import csv
import json
import logging
from pathlib import Path
from typing import Any, Dict

import cv2
import numpy as np


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

    frame_number = int(filename.split("_")[-1].split(".")[0])
    return {
        "frame_number": frame_idx,
        "x": x,
        "y": y,
        "width": width,
        "height": height,
        "id": track_id,
    }


def write_tracked_bbox_to_csv(
    bbox: np.ndarray,
    frame: np.ndarray,
    frame_name: str,
    csv_writer: Any,
    pred_score: np.ndarray,
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
    pred_score : np.ndarray
        The prediction score from detector.
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
            '{{"track":"{}", "confidence":"{}"}}'.format(int(id), pred_score),
        )
    )


def save_output_frames(
    frame_name: str,
    tracking_output_dir: Path,
    frame: np.ndarray,
    frame_number: int,
) -> None:
    """
    Save tracked bounding boxes as frames.

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
    pred_scores : np.ndarray
        The prediction score from detector

    Returns
    -------
    None
    """

    # Save frame as PNG - once as per frame
    frame_path = tracking_output_dir / frame_name
    img_saved = cv2.imwrite(str(frame_path), frame)
    if not img_saved:
        logging.error(
            f"Didn't save {frame_name}, frame {frame_number}, Skipping."
        )


def prep_sort(prediction: dict, score_threshold: float) -> np.ndarray:
    """
    Put predictions in format expected by SORT

    Parameters
    ----------
    prediction : dict
        The dictionary containing predicted bounding boxes, scores, and labels.

    Returns
    -------
    np.ndarray:
        An array containing sorted bounding boxes of detected objects.
    """
    pred_boxes = prediction[0]["boxes"].detach().cpu().numpy()
    pred_scores = prediction[0]["scores"].detach().cpu().numpy()
    pred_labels = prediction[0]["labels"].detach().cpu().numpy()

    pred_sort = []
    for box, score, label in zip(pred_boxes, pred_scores, pred_labels):
        if score > score_threshold:
            bbox = np.concatenate((box, [score]))
            pred_sort.append(bbox)

    return np.asarray(pred_sort)


def get_predicted_data(predicted_boxes_id) -> Dict[int, Dict[str, Any]]:
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

    for frame_idx, frame_data in enumerate(predicted_boxes_id):
        if frame_data.size == 0:
            continue

        bboxes = frame_data[:, :4]
        ids = frame_data[:, 4]

        predicted_dict[frame_idx] = {"bbox": bboxes, "id": ids}

    return predicted_dict


def get_ground_truth_data(gt_dir) -> Dict[int, Dict[str, Any]]:
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
    with open(gt_dir, "r") as csvfile:
        csvreader = csv.reader(csvfile)
        next(csvreader)  # Skip the header row
        ground_truth_data = [
            extract_bounding_box_info(row) for row in csvreader
        ]

    # Format as a dictionary with key = frame number
    ground_truth_dict: dict = {}
    for data in ground_truth_data:
        frame_idx = data["frame_number"]
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

        if frame_idx not in ground_truth_dict:
            ground_truth_dict[frame_idx] = {"bbox": [], "id": []}

        ground_truth_dict[frame_idx]["bbox"].append(bbox)
        ground_truth_dict[frame_idx]["id"].append(track_id)

        # format as numpy arrays
    for frame_idx in ground_truth_dict:
        ground_truth_dict[frame_idx]["bbox"] = np.array(
            ground_truth_dict[frame_idx]["bbox"], dtype=np.float32
        )
        ground_truth_dict[frame_idx]["id"] = np.array(
            ground_truth_dict[frame_idx]["id"], dtype=np.float32
        )
    return ground_truth_dict
