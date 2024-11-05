"""Utility functions for handling input and output operations."""

import csv
import logging

import cv2
import numpy as np

from crabs.detector.utils.visualization import draw_bbox


def get_video_parameters(video: cv2.VideoCapture) -> dict:
    """Get total number of frames, frame width and height, and fps of video."""
    video_parameters = {}
    video_parameters["total_frames"] = int(video.get(cv2.CAP_PROP_FRAME_COUNT))
    video_parameters["frame_width"] = int(video.get(cv2.CAP_PROP_FRAME_WIDTH))
    video_parameters["frame_height"] = int(
        video.get(cv2.CAP_PROP_FRAME_HEIGHT)
    )
    video_parameters["fps"] = video.get(cv2.CAP_PROP_FPS)
    return video_parameters


def write_tracked_detections_to_csv(
    csv_file_path: str,
    tracked_bboxes_per_frame: list[np.ndarray],
    pred_bboxes_scores_per_frame: list[np.ndarray],
    frame_name_regexp: str = "frame_{frame_idx:08d}.png",
    all_frames_size: int = 8888,
):
    """Write tracked detections to a csv file."""
    # Initialise csv file
    csv_file = open(  # noqa: SIM115
        csv_file_path,
        "w",
    )
    csv_writer = csv.writer(csv_file)

    # write header following VIA convention
    # https://www.robots.ox.ac.uk/~vgg/software/via/docs/face_track_annotation.html
    csv_writer.writerow(
        (
            "filename",
            "file_size",
            "file_attributes",
            "region_count",
            "region_id",
            "region_shape_attributes",
            "region_attributes",
        )
    )

    # write detections
    for frame_idx in range(len(tracked_bboxes_per_frame)):
        for bbox, pred_score in zip(
            tracked_bboxes_per_frame[frame_idx],
            pred_bboxes_scores_per_frame[frame_idx],
        ):
            xmin, ymin, xmax, ymax, id = bbox
            width_box = int(xmax - xmin)
            height_box = int(ymax - ymin)

            # Add to csv
            csv_writer.writerow(
                (
                    frame_name_regexp.format(
                        frame_idx=frame_idx
                    ),  # f"frame_{frame_idx:08d}.png",  # frame index!
                    all_frames_size,  # frame size
                    '{{"clip":{}}}'.format("123"),
                    1,
                    0,
                    f'{{"name":"rect","x":{xmin},"y":{ymin},"width":{width_box},"height":{height_box}}}',
                    f'{{"track":"{int(id)}", "confidence":"{pred_score}"}}',
                )
            )


def write_frame_to_output_video(
    frame: np.array,
    tracked_boxes_id_per_frame: list,
    output_video_object: cv2.VideoWriter,
) -> None:
    """Write frame with tracked bounding boxes to output video."""
    frame_copy = frame.copy()  # why copy?
    for bbox in tracked_boxes_id_per_frame:
        xmin, ymin, xmax, ymax, id = bbox

        draw_bbox(
            frame_copy,
            (xmin, ymin),
            (xmax, ymax),
            (0, 0, 255),
            f"id : {int(id)}",
        )
    output_video_object.write(frame_copy)


def write_frame_as_image(frame, frame_path):
    """Write frame as image without detections."""
    img_saved = cv2.imwrite(
        frame_path,
        frame,
    )
    if not img_saved:
        logging.error(
            f"Error saving {frame_path}."  # f"frame_{frame_idx:08d}.png"
        )
