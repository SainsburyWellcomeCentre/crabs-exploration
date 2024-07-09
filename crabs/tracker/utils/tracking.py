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
        "frame_number": frame_number,
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
