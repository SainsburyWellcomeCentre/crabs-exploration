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

    frame_number = int(filename.split("_")[-1].split(".")[0]) - 1
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
    theta: float,
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
            '{{"track":"{}", "theta":"{}"}}'.format(int(id), theta),
        )
    )


def save_frame_and_csv(
    frame_name: str,
    tracking_output_dir: Path,
    tracked_boxes: list[list[float]],
    frame: np.ndarray,
    frame_number: int,
    csv_writer: Any,
    theta_list: list[float],
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
    theta_list: list[float]
        List of orientation for each bounding box

    Returns
    -------
    None
    """
    for bbox, theta in zip(tracked_boxes, theta_list):
        # Add bbox to csv
        write_tracked_bbox_to_csv(bbox, frame, frame_name, csv_writer, theta)

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


def calculate_velocity(tracked_boxes, previous_positions, frame_time_interval):
    velocities = []
    for track_box in tracked_boxes:
        track_id = int(track_box[4])  # track ID
        x_min, y_min, x_max, y_max = track_box[:4]
        cx, cy = (x_min + x_max) / 2, (
            y_min + y_max
        ) / 2  # center of the bounding box

        if track_id in previous_positions:
            prev_cx, prev_cy = previous_positions[track_id]
            # distance between current centre to the previous one
            dx = cx - prev_cx
            dy = cy - prev_cy
            # velocity = distance/time
            vx = dx / frame_time_interval
            vy = dy / frame_time_interval
            velocities.append((track_id, vx, vy))

        # Update previous positions
        previous_positions[track_id] = (cx, cy)

    return velocities


def get_orientation(tracked_boxes, velocities):
    orientation_data = (
        {}
    )  # Dictionary to store theta and arrow endpoints for each track_id

    for track_box, (track_id, vx, vy) in zip(tracked_boxes, velocities):
        x_min, y_min, x_max, y_max, _ = track_box
        cx, cy = (x_min + x_max) / 2, (
            y_min + y_max
        ) / 2  # Center of the bounding box

        # Calculate orientation angle in radians from velocity components
        if vx != 0 or vy != 0:
            theta = np.arctan2(vy, vx)
        else:
            theta = 0

        # Calculate arrow endpoints
        arrow_length = 50  # Length of the arrow in pixels
        end_x = int(cx + arrow_length * np.cos(theta))
        end_y = int(cy + arrow_length * np.sin(theta))

        # Store theta and arrow endpoints in the dictionary with track_id as key
        orientation_data[track_id] = {
            "theta": theta,
            "end_x": end_x,
            "end_y": end_y,
        }

    return orientation_data


# def get_orientation(tracked_boxes, velocities):
#     theta_list = []
#     for track_box, (track_id, vx, vy) in zip(tracked_boxes, velocities):
#         x_min, y_min, x_max, y_max, _ = track_box
#         cx, cy = (x_min + x_max) / 2, (
#             y_min + y_max
#         ) / 2  # center of the bounding box

#         # Calculate orientation angle in radians from velocity components
#         if vx != 0 or vy != 0:
#             theta = np.arctan2(vy, vx)
#         else:
#             theta = 0
#         theta_list.append(theta)

#         # # for visualisation for now
#         # # Calculate arrow endpoints
#         # arrow_length = 50  # Length of the arrow in pixels
#         # end_x = int(cx + arrow_length * np.cos(theta))
#         # end_y = int(cy + arrow_length * np.sin(theta))

#         # # Draw arrow on the frame
#         # cv2.arrowedLine(
#         #     frame, (int(cx), int(cy)), (end_x, end_y), (0, 255, 0), 2
#         # )

#         # # Optionally, draw bounding box and object ID
#         # cv2.rectangle(
#         #     frame,
#         #     (int(x_min), int(y_min)),
#         #     (int(x_max), int(y_max)),
#         #     (0, 255, 0),
#         #     2,
#         # )
#         # cv2.putText(
#         #     frame,
#         #     f"ID: {int(track_box[4])}",
#         #     (int(x_min), int(y_min) - 10),
#         #     cv2.FONT_HERSHEY_SIMPLEX,
#         #     0.5,
#         #     (0, 255, 0),
#         #     2,
#         # )

#     return theta_list
