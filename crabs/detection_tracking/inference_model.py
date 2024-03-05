import argparse
import csv
import logging
import os
from pathlib import Path

import cv2
import numpy as np
import torch
import torchvision.transforms as transforms
from sort import Sort

from crabs.detection_tracking.detection_utils import draw_bbox


class MyDialect(csv.Dialect):
    delimiter = ","  # Set your delimiter
    quotechar = '"'  # Set your quote character
    # lineterminator =
    quoting = csv.QUOTE_MINIMAL  # Set quoting behavior


class DetectorInference:
    """
    A class for performing object detection or tracking inference on a video
    using a pre-trained model.

    Parameters:
        args (argparse.Namespace): Command-line arguments containing
        configuration settings.

    Attributes:
        args (argparse.Namespace): The command-line arguments provided.
        vid_dir (str): The path to the input video.
        iou_threshold (float): The iou threshold for tracking.
        score_threshold (float): The score confidence threshold for tracking.
        sort_crab (Sort): An instance of the sorting algorithm used for tracking.
    """

    def __init__(self, args: argparse.Namespace) -> None:
        self.args = args
        self.vid_dir = args.vid_dir
        self.score_threshold = args.score_threshold
        self.iou_threshold = args.iou_threshold
        self.sort_tracker = Sort(
            max_age=args.max_age,
            min_hits=args.min_hits,
            iou_threshold=self.iou_threshold,
        )
        self.video_file_root = f"{Path(self.vid_dir).stem}_"

    def load_trained_model(self) -> torch.nn.Module:
        """
        Load the trained model.

        Returns
        -------
        None
        """
        model = torch.load(
            self.args.model_dir,
            map_location=torch.device(self.args.accelerator),
        )
        model.eval()
        return model

    def prep_sort(self, prediction):
        """
        Track objects in the predicted bounding boxes.

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
            if label == 1 and score > self.score_threshold:
                bbox = np.concatenate((box, [score]))
                pred_sort.append(bbox)

        return np.asarray(pred_sort)

    def load_video(self) -> None:
        """
        Load the input video and and prepare the output video.
        """
        self.trained_model = self.load_trained_model()

        self.video = cv2.VideoCapture(self.vid_dir)
        if not self.video.isOpened():
            raise Exception("Error opening video file")

        frame_width = int(self.video.get(cv2.CAP_PROP_FRAME_WIDTH))
        frame_height = int(self.video.get(cv2.CAP_PROP_FRAME_HEIGHT))
        cap_fps = self.video.get(cv2.CAP_PROP_FPS)

        if self.args.save_video:
            output_file = f"{self.video_file_root}output_video.mp4"
            output_codec = cv2.VideoWriter_fourcc(*"mp4v")
            self.out = cv2.VideoWriter(
                output_file, output_codec, cap_fps, (frame_width, frame_height)
            )

    def run_inference(self):
        """
        Run object detection or tracking inference on the video frames.
        """
        transform = transforms.Compose([transforms.ToTensor()])
        frame_number = 1

        if self.args.save_csv_and_frames:
            tracking_output_dir = Path("tracking_output")
            tracking_output_dir.mkdir(parents=True, exist_ok=True)

            csv_file = open(
                str(tracking_output_dir / "tracking_output.csv"), "w"
            )
            csv_writer = csv.writer(csv_file)
            # dialect=MyDialect, lineterminator="\r\n")
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

        while self.video.isOpened():
            if frame_number > 1:
                break
            ret, frame = self.video.read()
            if not ret:
                print("No frame read. Exiting...")
                break
            # pdb.set_trace()
            if self.args.save_video:
                frame_copy = frame.copy()

            img = transform(frame).to(self.args.accelerator)
            img = img.unsqueeze(0)
            prediction = self.trained_model(img)
            pred_sort = self.prep_sort(prediction)

            tracked_boxes = self.sort_tracker.update(pred_sort)

            for bbox in tracked_boxes:
                xmin, ymin, xmax, ymax, id = bbox
                id_label = f"id : {int(id)}"
                if self.args.save_video:
                    draw_bbox(
                        frame_copy,
                        int(xmin),
                        int(ymin),
                        int(xmax),
                        int(ymax),
                        (0, 0, 255),
                        id_label,
                    )
                if self.args.save_csv_and_frames:
                    frame_name = (
                        f"{self.video_file_root}frame_{frame_number:08d}.png"
                    )

                    width_box = int(xmax - xmin)
                    height_box = int(ymax - ymin)
                    # quoted_columns = [2, 4, 5]
                    # quoted_row = ['"{}"'.format(cell) if i in quoted_columns else cell for i, cell in enumerate(row)]
                    csv_writer.writerow(
                        (
                            frame_name,
                            frame.size,
                            # str({"\"clip\"": 123}),
                            # '"{}"'.format({"clip": 123}),
                            {'"clip"': 123},
                            1,
                            0,
                            {
                                "name": "rect",
                                "x": int(xmin),
                                "y": int(ymin),
                                "width": width_box,
                                "height": height_box,
                            },
                            {"track": int(id)},
                        )
                    )

                    file_path = tracking_output_dir / frame_name
                    img_saved = cv2.imwrite(str(file_path), frame)
                    if img_saved:
                        logging.info(
                            f"frame {frame_number} saved at {file_path}"
                        )
                    else:
                        logging.info(
                            f"ERROR saving {frame_name}, frame {frame_number}"
                            "...skipping",
                        )
                        continue

            frame_number += 1

            if self.args.save_video:
                self.out.write(frame_copy)

        self.video.release()
        if self.args.save_video:
            self.out.release()
        if args.save_csv_and_frames:
            csv_file.close()


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
        "--vid_dir",
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
        default=1,
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
        help="save predicted tracks in VIA csv format.",
    )
    args = parser.parse_args()
    main(args)
