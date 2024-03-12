import argparse
import csv
import json
import logging
import os
from pathlib import Path

import cv2
import numpy as np
import torch
import torchvision.transforms as transforms
from sort import Sort

from crabs.detection_tracking.detection_utils import calculate_iou, draw_bbox


def evaluate_tracking(gt_boxes_list, tracked_boxes_list, iou_threshold):
    mota_values = []
    for gt_boxes, tracked_boxes in zip(gt_boxes_list, tracked_boxes_list):
        mota = evaluate_mota(gt_boxes, tracked_boxes, iou_threshold)
        mota_values.append(mota)
    return mota_values


def evaluate_mota(gt_boxes, tracked_boxes, iou_threshold):
    total_gt = len(gt_boxes)
    false_alarms = 0

    # List to store indices of tracked boxes to remove
    indices_to_remove = []

    for i, tracked_box in enumerate(tracked_boxes):
        best_iou = 0
        best_match = None

        for j, gt_box in enumerate(gt_boxes):
            iou = calculate_iou(gt_box[:4], tracked_box[:4])
            if iou > iou_threshold and iou > best_iou:
                best_iou = iou
                best_match = j
        if best_match is not None:
            gt_boxes[best_match] = None
            indices_to_remove.append(i)
        else:
            false_alarms += 1

    # Remove tracked boxes marked for removal
    tracked_boxes = np.delete(tracked_boxes, indices_to_remove, axis=0)

    missed_detections = 0
    for box in gt_boxes:
        if box is not None and not np.all(np.isnan(box)):
            missed_detections += 1

    num_switches = count_identity_switches(gt_boxes, tracked_boxes)
    mota = 1 - (missed_detections + false_alarms + num_switches) / total_gt
    return mota


def count_identity_switches(ids_prev_frame, ids_current_frame):
    """
    Count the number of identity switches between two sets of object IDs.
    """
    # Convert NumPy arrays to tuples
    ids_prev_frame_tuples = [tuple(box) for box in ids_prev_frame]
    ids_current_frame_tuples = [tuple(box) for box in ids_current_frame]

    # Create dictionaries to track object IDs in each frame
    id_to_index_prev = {id_: i for i, id_ in enumerate(ids_prev_frame_tuples)}
    id_to_index_current = {
        id_: i for i, id_ in enumerate(ids_current_frame_tuples)
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


class DetectorInference:
    """
    A class for performing object detection or tracking inference on a video
    using a trained model.

    Parameters:
        args (argparse.Namespace): Command-line arguments containing
        configuration settings.

    Attributes:
        args (argparse.Namespace): The command-line arguments provided.
        vid_path (str): The path to the input video.
        iou_threshold (float): The iou threshold for tracking.
        score_threshold (float): The score confidence threshold for tracking.
        sort_tracker (Sort): An instance of the sorting algorithm used for tracking.
    """

    def __init__(self, args: argparse.Namespace) -> None:
        self.args = args
        self.vid_path = args.vid_path
        self.score_threshold = args.score_threshold
        self.iou_threshold = args.iou_threshold
        self.sort_tracker = Sort(
            max_age=args.max_age,
            min_hits=args.min_hits,
            iou_threshold=self.iou_threshold,
        )
        self.video_file_root = f"{Path(self.vid_path).stem}_"
        self.tracking_output_dir = Path("tracking_output")

    def load_trained_model(self) -> torch.nn.Module:
        """
        Load the trained model.

        Returns
        -------
        torch.nn.Module
        """
        model = torch.load(
            self.args.model_dir,
            map_location=torch.device(self.args.accelerator),
        )
        model.eval()
        return model

    def prep_sort(self, prediction):
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
            if score > self.score_threshold:
                bbox = np.concatenate((box, [score]))
                pred_sort.append(bbox)

        return np.asarray(pred_sort)

    def load_video(self) -> None:
        """
        Load the input video, and prepare the output video if required.
        """
        # load trained model
        self.trained_model = self.load_trained_model()

        # load input video
        self.video = cv2.VideoCapture(self.vid_path)
        if not self.video.isOpened():
            raise Exception("Error opening video file")

        # read input video parameters
        frame_width = int(self.video.get(cv2.CAP_PROP_FRAME_WIDTH))
        frame_height = int(self.video.get(cv2.CAP_PROP_FRAME_HEIGHT))
        cap_fps = self.video.get(cv2.CAP_PROP_FPS)

        # prepare output video writer if required
        if self.args.save_video:
            output_file = f"{self.video_file_root}output_video.mp4"
            output_codec = cv2.VideoWriter_fourcc(*"H264")
            self.out = cv2.VideoWriter(
                output_file, output_codec, cap_fps, (frame_width, frame_height)
            )

    def prep_csv_writer(self):
        """
        Prepare csv writer to output tracking results
        """
        self.tracking_output_dir.mkdir(parents=True, exist_ok=True)

        csv_file = open(
            str(self.tracking_output_dir / "tracking_output.csv"), "w"
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

    def write_bbox_to_csv(self, bbox, frame, frame_name, csv_writer):
        """
        Write bounding box annotation to csv
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
                '{{"track":{}}}'.format(id),
            )
        )

    # Common functionality for saving frames and CSV
    def save_frame_and_csv(
        self, tracked_boxes, frame, frame_number, csv_writer, save_plot=True
    ):
        frame_copy = frame.copy()

        for bbox in tracked_boxes:
            # Get frame name
            frame_name = f"{self.video_file_root}frame_{frame_number:08d}.png"

            # Add bbox to csv
            self.write_bbox_to_csv(bbox, frame, frame_name, csv_writer)

            # Save frame as PNG
            frame_path = self.tracking_output_dir / frame_name
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

    def get_ground_truth_data(self):
        # Initialize a list to store the extracted data
        ground_truth_data = []
        max_frame_number = 0

        # Open the CSV file and read its contents line by line
        with open(self.args.gt_dir, "r") as csvfile:
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
                ]
            )
            gt_boxes_list[frame_number] = (
                np.vstack([gt_boxes_list[frame_number], bbox])
                if gt_boxes_list[frame_number].size
                else bbox
            )

        return gt_boxes_list

    def run_inference(self):
        """
        Run object detection + tracking on the video frames.
        """
        # Get transform to tensor
        transform = transforms.Compose([transforms.ToTensor()])

        # initialise frame counter
        frame_number = 1

        # initialise csv writer if required
        if self.args.save_csv_and_frames:
            csv_writer, csv_file = self.prep_csv_writer()

        if self.args.gt_dir:
            gt_boxes_list = self.get_ground_truth_data()
            tracked_list = []

        # loop thru frames of clip
        while self.video.isOpened():
            # break if beyond end frame (mostly for debugging)
            if self.args.max_frames_to_read:
                if frame_number > self.args.max_frames_to_read:
                    break

            # read frame
            ret, frame = self.video.read()
            if not ret:
                print("No frame read. Exiting...")
                break

            # run prediction
            img = transform(frame).to(self.args.accelerator)
            img = img.unsqueeze(0)
            prediction = self.trained_model(img)

            # run tracking
            pred_sort = self.prep_sort(prediction)
            tracked_boxes = self.sort_tracker.update(pred_sort)
            # print(tracked_boxes.shape)
            tracked_list.append(tracked_boxes)

            if self.args.save_csv_and_frames:
                if self.args.save_video:
                    frame_copy = self.save_frame_and_csv(
                        tracked_boxes, frame, frame_number, csv_writer
                    )
                    self.out.write(frame_copy)
                else:
                    self.save_frame_and_csv(
                        tracked_boxes,
                        frame,
                        frame_number,
                        csv_writer,
                        save_plot=False,
                    )
            elif self.args.save_video:
                frame_copy = frame.copy()
                for bbox in tracked_boxes:
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
                self.out.write(frame_copy)

            # update frame
            frame_number += 1

        if self.args.gt_dir:
            mota_values = evaluate_tracking(
                gt_boxes_list, tracked_list, self.iou_threshold
            )
            overall_mota = np.mean(mota_values)
            print("Overall MOTA:", overall_mota)

        # # Close input video
        self.video.release()

        # Close outputs
        if self.args.save_video:
            self.out.release()

        if args.save_csv_and_frames:
            csv_file.close()
        # cv2.destroyAllWindows()


def main(args) -> None:
    """
    Main function to run the inference on video based on the trained model.

    Parameters
    ----------
    args : argparse
        Arguments or configuration settings for testing.

    Returns
    -------
        None
    """

    inference = DetectorInference(args)
    inference.load_video()
    inference.run_inference()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model_dir",
        type=str,
        required=True,
        help="location of trained model",
    )
    parser.add_argument(
        "--vid_path",
        type=str,
        required=True,
        help="location of images and coco annotation",
    )
    parser.add_argument(
        "--save_video",
        action="store_true",
        help="save video inference",
    )
    parser.add_argument(
        "--output_path",
        type=str,
        default=os.getcwd(),
        help="location of output video",
    )
    parser.add_argument(
        "--score_threshold",
        type=float,
        default=0.1,
        help="threshold for prediction score",
    )
    parser.add_argument(
        "--iou_threshold",
        type=float,
        default=0.1,
        help="threshold for prediction score",
    )
    parser.add_argument(
        "--max_age",
        type=int,
        default=10,
        help="Maximum number of frames to keep alive a track without associated detections.",
    )
    parser.add_argument(
        "--min_hits",
        type=int,
        default=1,
        help="Minimum number of associated detections before track is initialised.",
    )
    parser.add_argument(
        "--accelerator",
        type=str,
        default="gpu",
        help="accelerator for pytorch lightning",
    )
    parser.add_argument(
        "--save_csv_and_frames",
        action="store_true",
        help=(
            "Save predicted tracks in VIA csv format and export corresponding frames. "
            "This is useful to prepare for manual labelling of tracks."
        ),
    )
    parser.add_argument(
        "--max_frames_to_read",
        type=int,
        default=None,
        help="Maximum number of frames to read (mostly for debugging).",
    )
    parser.add_argument(
        "--gt_dir",
        type=str,
        default=None,
        help="Directory contains ground truth annotations.",
    )
    args = parser.parse_args()
    main(args)
