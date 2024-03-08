import argparse
import csv
import json
import logging
import os
from pathlib import Path
import pandas as pd
import cv2
import numpy as np
import torch
import torchvision
import torchvision.transforms as transforms
from sort import Sort

from crabs.detection_tracking.detection_utils import draw_bbox


def apply_nms(prediction, threshold=0.1):
    """
    Apply Non-Maximum Suppression (NMS) to a single prediction dictionary.
    
    Args:
        prediction: Dictionary containing 'boxes', 'labels', and 'scores' tensors.
        threshold: IoU threshold for NMS.
        
    Returns:
        Dictionary containing filtered bounding boxes, labels, and scores after NMS.
    """
    boxes = [pred['boxes'] for pred in prediction]
    scores = [pred['scores'] for pred in prediction]

    # Apply NMS
    nms_threshold = 0.2  # You can adjust this threshold as needed
    keep_indices = torchvision.ops.nms(boxes[0], scores[0], nms_threshold)

    # Select only the boxes, labels, and scores that survived NMS
    filtered_boxes = prediction[0]['boxes'][keep_indices]
    filtered_labels = prediction[0]['labels'][keep_indices]
    filtered_scores = prediction[0]['scores'][keep_indices]

    filtered_prediction = [{'boxes': filtered_boxes,
                            'labels': filtered_labels,
                            'scores': filtered_scores}]

    return filtered_prediction


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
        if self.args.save_video or self.args.gt_dir:
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

                # Append the extracted data to the list
                ground_truth_data.append(
                    {
                        "filename": filename,
                        "frame_number": frame_number,
                        "x": x,
                        "y": y,
                        "width": width,
                        "height": height,
                        "id": track_id,
                    }
                )
        return ground_truth_data
    

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
            from crabs.detection_tracking.detection_utils import (
                draw_gt_tracking,
            )
            # from trackeval.metrics import HOTA

            ground_truth_data = self.get_ground_truth_data()
            # # Define the HOTA metric
            # hota_metric = HOTA()

            # all_sequence_results = []

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
            # print(prediction)

            # perform Non-Maximum Suppression (NMS)
            nms_prediction = apply_nms(prediction)

            # run tracking
            pred_sort = self.prep_sort(nms_prediction)
            tracked_boxes = self.sort_tracker.update(pred_sort)

            if self.args.gt_dir:
                gt_boxes = []
                for gt_data in ground_truth_data:
                    if gt_data["frame_number"] == frame_number:
                        gt_boxes.append(
                            (
                                gt_data["x"],
                                gt_data["y"],
                                gt_data["x"] + gt_data["width"],
                                gt_data["y"] + gt_data["height"],
                                gt_data["id"],
                            )
                        )
                gt_boxes = np.asarray(gt_boxes)

                # # Evaluate HOTA metric for the current frame
                # # sequence_data = (tracked_boxes, gt_boxes)
                # sequence_data = {
                #     'num_tracker_ids': len(np.unique(tracked_boxes[:, -1])),
                #     'num_gt_ids': len(np.unique(gt_boxes[:, -1])),
                #     'tracker_ids': np.array(tracked_boxes[:, -1], dtype=np.float64),  # Convert to numpy array and set data type
                #     'gt_ids': np.array(gt_boxes[:, -1], dtype=np.float64),  # Convert to numpy array and set data type
                #     'similarity_scores': np.random.rand(len(gt_boxes), len(tracked_boxes)).astype(np.float64),  # Placeholder for similarity scores with explicit data type
                # }
                # print(sequence_data)
                # sequence_results = hota_metric.eval_sequence(sequence_data)
                # all_sequence_results.append(sequence_results)

                # # frame_copy = frame.copy()
                # # frame_copy = draw_gt_tracking(
                # #     gt_boxes,
                # #     tracked_boxes,
                # #     frame_number,
                # #     self.iou_threshold,
                # #     frame_copy,
                # # )
                # # self.out.write(frame_copy)
                # # cv2.imshow("frame", frame_copy)
                # # if cv2.waitKey(30) & 0xFF == 27:
                # #     break

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

        # if self.args.gt_dir:
        #     # Combine results across all sequences
        #     combined_results = hota_metric.combine_sequences(all_sequence_results)
        #     print(combined_results)
        # # Close input video
        self.video.release()

        # Close outputs
        if self.args.save_video or self.args.gt_dir:
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
        default=5,
        help="Maximum number of frames to keep alive a track without associated detections.",
    )
    parser.add_argument(
        "--min_hits",
        type=int,
        default=3,
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
