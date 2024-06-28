import csv
import os
from pathlib import Path

import cv2
import numpy as np

from crabs.detector.utils.visualization import draw_bbox
from crabs.tracker.utils.tracking import (
    save_frame_and_csv,
    write_tracked_bbox_to_csv,
)


def prep_csv_writer(output_dir, video_file_root):
    """
    Prepare csv writer to output tracking results
    """

    crabs_tracks_label_dir = Path(output_dir) / "crabs_tracks_label"
    tracking_output_dir = crabs_tracks_label_dir / video_file_root
    # Create the subdirectory for the specific video file root
    tracking_output_dir.mkdir(parents=True, exist_ok=True)

    csv_file = open(
        f"{str(tracking_output_dir / video_file_root)}.csv",
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

    return csv_writer, csv_file


def prep_video_writer(
    output_dir, video_file_root, frame_width, frame_height, cap_fps
):
    # create directory to save output
    os.makedirs(output_dir, exist_ok=True)

    output_file = os.path.join(
        output_dir,
        f"{os.path.basename(video_file_root)}_output_video.mp4",
    )
    output_codec = cv2.VideoWriter_fourcc("m", "p", "4", "v")
    video_output = cv2.VideoWriter(
        output_file, output_codec, cap_fps, (frame_width, frame_height)
    )

    return video_output


def save_required_output(
    video_file_root,
    save_csv_and_frames,
    tracking_output_dir,
    csv_writer,
    save_video,
    video_output,
    tracked_boxes: list[list[float]],
    frame: np.ndarray,
    frame_number: int,
) -> None:
    """
    Handle the output based argument options.

    Parameters
    ----------
    tracked_boxes : list[list[float]]
        list of tracked bounding boxes.
    frame : np.ndarray
        The current frame.
    frame_number : int
        The frame number.
    """
    frame_name = f"{video_file_root}_frame_{frame_number:08d}.png"
    if save_csv_and_frames:
        save_frame_and_csv(
            frame_name,
            tracking_output_dir,
            tracked_boxes,
            frame,
            frame_number,
            csv_writer,
        )
    else:
        for bbox in tracked_boxes:
            write_tracked_bbox_to_csv(bbox, frame, frame_name, csv_writer)

    if save_video:
        frame_copy = frame.copy()
        for bbox in tracked_boxes:
            xmin, ymin, xmax, ymax, id = bbox
            draw_bbox(
                frame_copy,
                (xmin, ymin),
                (xmax, ymax),
                (0, 0, 255),
                f"id : {int(id)}",
            )
        video_output.write(frame_copy)


def close_csv_file(csv_file) -> None:
    """
    Close the CSV file if it's open.
    """
    if csv_file:
        csv_file.close()


def release_video(video_output) -> None:
    """
    Release the video file if it's open.
    """
    if video_output:
        video_output.release()
