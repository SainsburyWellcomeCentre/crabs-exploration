import argparse
import os
from pathlib import Path

import cv2
import torch
import torchvision.transforms as transforms
from sort import Sort

# select device (whether GPU or CPU)
device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")


class Detector_Test:
    def __init__(self, args: argparse.Namespace) -> None:
        self.args = args
        self.vid_dir = args.vid_dir
        self.score_threshold = args.score_threshold
        self.sort_crab = Sort()

    def _load_pretrain_model(self) -> None:
        """Load the pretrain subject classification model"""

        # Load the pretrain subject predictor
        # TODO:deal with different model
        self.trained_model = torch.load(
            self.args.model_dir, map_location=torch.device("cpu")
        )

    def __inference(self, frame, video_file, frame_id):
        self.trained_model.eval()
        transform = transforms.Compose(
            [
                transforms.ToTensor(),
            ]
        )
        img = transform(frame)
        img = img.to(device)

        img = img.unsqueeze(0)
        prediction = self.trained_model(img)
        pred_score = list(prediction[0]["scores"].detach().cpu().numpy())

        if not self.args.sort:
            from _inference import inference_detection

            inference_detection(frame, prediction, pred_score, self.score_threshold)

        else:
            from _inference import inference_tracking

            inference_tracking(
                frame, prediction, pred_score, self.score_threshold, self.sort_crab
            )

    def _load_video(self) -> None:
        """Load video for the inference"""
        video = cv2.VideoCapture(self.vid_dir)

        if not video.isOpened():
            raise Exception("Error opening video file")

        frame_width = int(video.get(cv2.CAP_PROP_FRAME_WIDTH))
        frame_height = int(video.get(cv2.CAP_PROP_FRAME_HEIGHT))
        cap_fps = video.get(cv2.CAP_PROP_FPS)

        output_file = "output_video.mp4"
        output_codec = cv2.VideoWriter_fourcc(*"mp4v")
        out = cv2.VideoWriter(
            output_file, output_codec, cap_fps, (frame_width, frame_height)
        )
        frame_id = 0

        video_file = f"{Path(self.vid_dir).parent.stem}_" f"{Path(self.vid_dir).stem}_"
        while video.isOpened():
            ret, frame = video.read()

            if not ret:
                # Break the loop if no more frames to read
                break

            frame_out = self.__inference(frame, video_file, frame_id)
            frame_id += 1
            out.write(frame_out)

        video.release()
        out.release()
        cv2.destroyAllWindows()

    def inference_model(self) -> None:
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
        default=False,
        help="running sort as tracker",
    )

    args = parser.parse_args()
    inference = Detector_Test(args)
    inference.inference_model()
