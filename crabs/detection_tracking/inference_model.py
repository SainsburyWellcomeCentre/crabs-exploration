import argparse
import os
from pathlib import Path

import cv2
import numpy as np
import torch
import torchvision.transforms as transforms
from sort import Sort
from crabs.detection_tracking.detection_utils import drawing_detection

# select device (whether GPU or CPU)
device = (
    torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
)


class Detector_Inference:
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

    Methods:
        _load_pretrain_model(self) -> None:
            Load the pre-trained subject classification model.

        __inference(self, frame, video_file, frame_id) -> None:
            Perform inference on a single frame of the video.

        _load_video(self) -> None:
            Load the input video and perform inference on its frames.

        inference_model(self) -> None:
            Perform object detection or tracking inference on the input video.

    """

    def __init__(self, args: argparse.Namespace) -> None:
        self.args = args
        self.vid_dir = args.vid_dir
        self.score_threshold = args.score_threshold
        self.sort_crab = Sort()

    def _load_pretrain_model(self) -> None:
        """
        Load the pre-trained subject classification model.
        """
        # Load the pre-trained subject predictor
        # TODO: deal with different model
        self.trained_model = torch.load(
            self.args.model_dir,
            map_location=torch.device("cpu")
        )

    # def __inference(self, frame: np.ndarray) -> np.ndarray:
    #     """
    #     Perform inference on a single frame of the video.

    #     Args:
    #         frame (np.ndarray): The input frame as a NumPy array.

    #     Returns:
    #         None
    #     """
    #     self.trained_model.eval()
    #     transform = transforms.Compose(
    #         [
    #             transforms.ToTensor(),
    #         ]
    #     )
    #     img = transform(frame)
    #     img = img.to(device)

    #     img = img.unsqueeze(0)
    #     prediction = self.trained_model(img)
    #     pred_score = list(prediction[0]["scores"].detach().cpu().numpy())

    #     # if not self.args.sort:
    #     #     from inference import inference_detection

    #     #     frame_out = inference_detection(
    #     #         frame, prediction, pred_score, self.score_threshold
    #     #     )

    #     # else:
    #     # from inference import inference_tracking

    #     frame_out = inference_tracking(
    #         frame,
    #         prediction,
    #         pred_score,
    #         self.score_threshold,
    #     )
    #     return frame_out

    def _load_video(self) -> None:
        """
        Load the input video and perform inference on its frames.
        """
        video = cv2.VideoCapture(self.vid_dir)

        if not video.isOpened():
            raise Exception("Error opening video file")

        frame_width = int(video.get(cv2.CAP_PROP_FRAME_WIDTH))
        frame_height = int(video.get(cv2.CAP_PROP_FRAME_HEIGHT))
        cap_fps = video.get(cv2.CAP_PROP_FPS)

        video_file = (
            f"{Path(self.vid_dir).parent.stem}_" f"{Path(self.vid_dir).stem}_"
        )

        output_file = f"{video_file}_output_video.mp4"
        output_codec = cv2.VideoWriter_fourcc(*"mp4v")
        out = cv2.VideoWriter(
            output_file, output_codec, cap_fps, (frame_width, frame_height)
        )

        self.trained_model.eval()
        transform = transforms.Compose(
            [
                transforms.ToTensor(),
            ]
        )
        
        while video.isOpened():
            ret, frame = video.read()
            if not ret:
                print("No frame read. Exiting...")
                break

            # frame_out = self.__inference(frame)
            img = transform(frame)
            img = img.to(device)

            img = img.unsqueeze(0)
            prediction = self.trained_model(img)
            frame_out = drawing_detection(
                img, prediction, score_threshold=0.5
            )
            out.write(frame_out)

            cv2.imshow("frame", frame_out)

            if cv2.waitKey(30) & 0xFF == 27:
                break

        video.release()
        out.release()
        cv2.destroyAllWindows()

    def inference_model(self) -> None:
        """
        Perform object detection or tracking inference on the input video.
        """
        self._load_pretrain_model()
        self._load_video()


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
        "--sort",
        type=bool,
        default=True,
        help="running sort as tracker",
    )

    args = parser.parse_args()
    inference = Detector_Inference(args)
    inference.inference_model()
