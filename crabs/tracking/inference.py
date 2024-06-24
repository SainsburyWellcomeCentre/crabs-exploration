import csv
import os
from pathlib import Path

import cv2
import numpy as np

from crabs.detection.visualization import draw_bbox
from crabs.tracking._utils import (
    save_frame_and_csv,
    write_tracked_bbox_to_csv,
)


class Inference:
    def __init__(
        self, output_dir, video_file_root, save_csv_and_frames, save_video
    ) -> None:
        self.output_dir = output_dir
        self.video_file_root = video_file_root
        self.save_csv_and_frames = save_csv_and_frames
        self.save_video = save_video
        self.prep_csv_writer()

    def prep_csv_writer(self) -> None:
        """
        Prepare csv writer to output tracking results
        """

        crabs_tracks_label_dir = Path(self.output_dir) / "crabs_tracks_label"
        self.tracking_output_dir = (
            crabs_tracks_label_dir / self.video_file_root
        )
        # Create the subdirectory for the specific video file root
        self.tracking_output_dir.mkdir(parents=True, exist_ok=True)

        self.csv_file = open(
            f"{str(self.tracking_output_dir / self.video_file_root)}.csv",
            "w",
        )
        self.csv_writer = csv.writer(self.csv_file)

        # write header following VIA convention
        # https://www.robots.ox.ac.uk/~vgg/software/via/docs/face_track_annotation.html
        self.csv_writer.writerow(
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

    def prep_video_writer(self, frame_width, frame_height, cap_fps):
        # create directory to save output
        os.makedirs(self.output_dir, exist_ok=True)

        output_file = os.path.join(
            self.output_dir,
            f"{os.path.basename(self.video_file_root)}_output_video.mp4",
        )
        output_codec = cv2.VideoWriter_fourcc("m", "p", "4", "v")
        self.video_output = cv2.VideoWriter(
            output_file, output_codec, cap_fps, (frame_width, frame_height)
        )

    def save_required_output(
        self,
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
        frame_name = f"{self.video_file_root}_frame_{frame_number:08d}.png"
        if self.save_csv_and_frames:
            save_frame_and_csv(
                frame_name,
                self.tracking_output_dir,
                tracked_boxes,
                frame,
                frame_number,
                self.csv_writer,
            )
        else:
            for bbox in tracked_boxes:
                write_tracked_bbox_to_csv(
                    bbox, frame, frame_name, self.csv_writer
                )

        if self.save_video:
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
            self.video_output.write(frame_copy)

    def close_csv_file(self) -> None:
        """
        Close the CSV file if it's open.
        """
        if self.csv_file:
            self.csv_file.close()

    def release_video(self) -> None:
        """
        Release the video file if it's open.
        """
        if self.video_output:
            self.video_output.release()
