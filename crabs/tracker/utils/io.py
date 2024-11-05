"""Utility functions for handling input and output operations."""

import argparse
import csv
import os
from datetime import datetime
from pathlib import Path

import cv2
import matplotlib.pyplot as plt
import numpy as np

from crabs.detector.utils.visualization import draw_bbox
from crabs.tracker.utils.tracking import (
    save_output_frame,
    write_tracked_bbox_to_csv,
)


def prep_csv_writer(output_dir: str, video_file_root: str):
    """Prepare csv writer to output tracking results.

    Parameters
    ----------
    output_dir : str
        The output folder where the output will be stored.
    video_file_root : str
        The root name of the video file.

    Returns
    -------
    Tuple
        A tuple containing the CSV writer, the CSV file object, and the
        tracking output directory path.

    """
    # Create a timestamped directory for the tracking output
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    tracking_output_dir = Path(output_dir + f"_{timestamp}")
    tracking_output_dir.mkdir(parents=True, exist_ok=True)

    # Initialise csv file
    csv_file = open(  # noqa: SIM115
        f"{str(tracking_output_dir)}/{video_file_root}_tracks.csv",
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

    return csv_writer, csv_file, tracking_output_dir


def prep_video_writer(
    output_dir: str,
    video_file_root: str,
    frame_width: int,
    frame_height: int,
    cap_fps: float,
) -> cv2.VideoWriter:
    """Prepare video writer to output processed video.

    Parameters
    ----------
    output_dir : str
        The output folder where the output will be stored.
    video_file_root :str
        The root name of the video file.
    frame_width : int
        The width of the video frames.
    frame_height : int
        The height of the video frames.
    cap_fps : float
        The frames per second of the video.

    Returns
    -------
    cv2.VideoWriter
        The video writer object for writing video frames.

    """
    output_file = os.path.join(
        output_dir,
        f"{video_file_root}_tracks.mp4",
    )
    output_codec = cv2.VideoWriter_fourcc("m", "p", "4", "v")
    video_output = cv2.VideoWriter(
        output_file, output_codec, cap_fps, (frame_width, frame_height)
    )

    return video_output


def save_required_output(
    video_file_root: Path,
    save_frames: bool,
    tracking_output_dir: Path,
    csv_writer: cv2.VideoWriter,
    save_video: bool,
    video_output: cv2.VideoWriter,
    tracked_boxes: list[list[float]],
    frame: np.ndarray,
    frame_number: int,
    pred_scores: np.ndarray,
) -> None:
    """Handle the output based on argument options.

    Parameters
    ----------
    video_file_root : Path
        The root name of the video file.
    save_frames : bool
        Flag to save frames.
    tracking_output_dir : Path
        Directory to save tracking output.
    csv_writer : Any
        CSV writer object.
    save_video : bool
        Flag to save video.
    video_output : cv2.VideoWriter
        Video writer object for writing video frames.
    tracked_boxes : list[list[float]]
        List of tracked bounding boxes.
    frame : np.ndarray
        The current frame.
    frame_number : int
        The frame number.
    pred_scores : np.ndarray
        The prediction score from detector

    """
    frame_name = f"frame_{frame_number:08d}.png"

    for bbox, pred_score in zip(tracked_boxes, pred_scores):
        write_tracked_bbox_to_csv(
            np.array(bbox), frame, frame_name, csv_writer, pred_score
        )

    if save_frames:
        # create subdirectory of frames
        frames_subdir = tracking_output_dir / f"{video_file_root}_frames"
        frames_subdir.mkdir(parents=True, exist_ok=True)

        # save frame (without bounding boxes)
        save_output_frame(
            frame_name,
            frames_subdir,
            frame,
            frame_number,
        )

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
    """Close the CSV file if it's open."""
    if csv_file:
        csv_file.close()


def release_video(video_output) -> None:
    """Release the video file if it's open."""
    if video_output:
        video_output.release()


def read_metrics_from_csv(filename):
    """Read the tracking output metrics from a CSV file.

    To be called by plot_output_histogram.

    Parameters
    ----------
    filename : str
        Name of the CSV file to read.

    Returns
    -------
    tuple:
        Tuple containing lists of true positives, missed detections,
        false positives, number of switches, and total ground truth
        for each frame.

    """
    true_positives_list = []
    missed_detections_list = []
    false_positives_list = []
    num_switches_list = []
    total_ground_truth_list = []
    mota_value_list = []

    with open(filename) as file:
        reader = csv.DictReader(file)
        for row in reader:
            true_positives_list.append(int(row["True Positives"]))
            missed_detections_list.append(int(row["Missed Detections"]))
            false_positives_list.append(int(row["False Positives"]))
            num_switches_list.append(int(row["Number of Switches"]))
            total_ground_truth_list.append(int(row["Total Ground Truth"]))
            mota_value_list.append(float(row["Mota"]))

    return (
        true_positives_list,
        missed_detections_list,
        false_positives_list,
        num_switches_list,
        total_ground_truth_list,
        mota_value_list,
    )


def plot_output_histogram(filename):
    """Plot metrics along with the total ground truth for each frame.

    Example usage:
    > filename = <video_name>/tracking_metrics_output.csv
    > python crabs/tracker/utils/io.py filename

    Parameters
    ----------
    filename : str
        Name of the CSV file to read.
    true_positives_list : list[int]
        List of counts of true positives for each frame.
    missed_detections_list : list[int]
        List of counts of missed detections for each frame.
    false_positives_list : list[int]
        List of counts of false positives for each frame.
    num_switches_list : list[int]
        List of counts of identity switches for each frame.
    total_ground_truth_list : list[int]
        List of total ground truth objects for each frame.

    """
    (
        true_positives_list,
        missed_detections_list,
        false_positives_list,
        num_switches_list,
        total_ground_truth_list,
        mota_value_list,
    ) = read_metrics_from_csv(filename)
    filepath = Path(filename)
    plot_name = filepath.name

    num_frames = len(true_positives_list)
    frame_numbers = range(1, num_frames + 1)

    plt.figure(figsize=(10, 6))

    overall_mota = sum(mota_value_list) / len(mota_value_list)

    # Calculate percentages
    true_positives_percentage = [
        tp / gt * 100 if gt > 0 else 0
        for tp, gt in zip(true_positives_list, total_ground_truth_list)
    ]
    missed_detections_percentage = [
        md / gt * 100 if gt > 0 else 0
        for md, gt in zip(missed_detections_list, total_ground_truth_list)
    ]
    false_positives_percentage = [
        fp / gt * 100 if gt > 0 else 0
        for fp, gt in zip(false_positives_list, total_ground_truth_list)
    ]
    num_switches_percentage = [
        ns / gt * 100 if gt > 0 else 0
        for ns, gt in zip(num_switches_list, total_ground_truth_list)
    ]

    # Plot metrics
    plt.plot(
        frames,
        true_positives_percentage,
        label=f"True Positives ({sum(true_positives_list)})",
        color="g",
    )
    plt.plot(
        frames,
        missed_detections_percentage,
        label=f"Missed Detections ({sum(missed_detections_list)})",
        color="r",
    )
    plt.plot(
        frames,
        false_positives_percentage,
        label=f"False Positives ({sum(false_positives_list)})",
        color="b",
    )
    plt.plot(
        frames,
        num_switches_percentage,
        label=f"Number of Switches ({sum(num_switches_list)})",
        color="y",
    )

    plt.xlabel("Frame Number")
    plt.ylabel("Percentage of Total Ground Truth (%)")
    plt.title(f"{plot_name}_mota:{overall_mota:.2f}")

    plt.legend()
    plt.savefig(f"{plot_name}.pdf")
    plt.show()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Plot output histogram.")
    parser.add_argument(
        "filename",
        type=str,
        help="Path to the CSV file containing the metrics",
    )
    args = parser.parse_args()
    plot_output_histogram(args.filename)
