import argparse
import os
from pathlib import Path

import cv2
import numpy as np
import torch
import torchvision.transforms as transforms
from sort import Sort
from crabs.detection_tracking.detection_utils import draw_bbox


class DetectorInference:
    """
    A class for performing object detection or tracking inference on a video
    using a pre-trained model.

    Args:
        args (argparse.Namespace): Command-line arguments containing
        configuration settings.

    Attributes:
        args (argparse.Namespace): The command-line arguments provided.
        vid_dir (str): The path to the input video.
        score_threshold (float): The confidence threshold for detection scores.
        sort_crab (Sort): An instance of the sorting algorithm used for tracking.
        trained_model: The pre-trained subject classification model.
    """

    def __init__(self, args: argparse.Namespace) -> None:
        self.args = args
        self.vid_dir = args.vid_dir
        self.score_threshold = args.score_threshold
        self.sort_tracker = Sort()

    def _load_trained_model(self) -> torch.nn.Module:
        """
        Load the trained model.

        Returns
        -------
        None
        """
        model = torch.load(self.args.model_dir, map_location=torch.device(self.args.accelerator))
        model.eval()
        return model
    
    def _track_objects(self, prediction):
        pred_boxes = prediction[0]["boxes"].detach().cpu().numpy()
        pred_scores = prediction[0]["scores"].detach().cpu().numpy()
        pred_labels = prediction[0]["labels"].detach().cpu().numpy()

        pred_sort = []
        for box, score, label in zip(pred_boxes, pred_scores, pred_labels):
            if label == 1 and score > self.score_threshold:  # Assuming label 1 represents "crab"
                bbox = np.concatenate((box, [score]))
                pred_sort.append(bbox)

        return np.asarray(pred_sort)
    
    def load_video(self) -> None:
        """
        Load the input video and and prepare the output video.
        """
        self.trained_model = self._load_trained_model()
        
        self.video = cv2.VideoCapture(self.vid_dir)
        if not self.video.isOpened():
            raise Exception("Error opening video file")

        frame_width = int(self.video.get(cv2.CAP_PROP_FRAME_WIDTH))
        frame_height = int(self.video.get(cv2.CAP_PROP_FRAME_HEIGHT))
        cap_fps = self.video.get(cv2.CAP_PROP_FPS)

        video_file = (
            f"{Path(self.vid_dir).parent.stem}_" f"{Path(self.vid_dir).stem}_"
        )

        output_file = f"{video_file}_output_video.mp4"
        output_codec = cv2.VideoWriter_fourcc(*"mp4v")
        self.out = cv2.VideoWriter(
            output_file, output_codec, cap_fps, (frame_width, frame_height)
        )

    def run_inference(self):

        transform = transforms.Compose([transforms.ToTensor()])

        while self.video.isOpened():
            ret, frame = self.video.read()
            if not ret:
                print("No frame read. Exiting...")
                break

            img = transform(frame).to(self.args.accelerator)
            img = img.unsqueeze(0)
            prediction = self.trained_model(img)
            pred_sort = self._track_objects(prediction)

            tracked_boxes = self.sort_tracker.update(pred_sort)

            for bbox in tracked_boxes:
                x1, y1, x2, y2, _ = bbox
                id_label = f"id : {bbox[4]}"
                draw_bbox(frame, int(x1), int(y1), int(x2), int(y2), (0, 0, 255), id_label)

            if self.args.save:
                self.out.write(frame)

            cv2.imshow("frame", frame)

            if cv2.waitKey(30) & 0xFF == 27:
                break

        self.video.release()
        self.out.release()
        cv2.destroyAllWindows()


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
        "--save",
        type=bool,
        default=True,
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
        default=0.5,
        help="threshold for prediction score",
    )
    parser.add_argument(
        "--accelerator",
        type=str,
        default="gpu",
        help="accelerator for pytorch lightning",
    )
    args = parser.parse_args()
    main(args)
