import argparse
import os
from pathlib import Path

import cv2
import numpy as np
import torch
import torchvision.transforms as transforms
from sort import Sort

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
            # map_location=torch.device("cpu")
        )

    def apply_grayscale_and_blur(
        self,
        frame: np.array,
        kernel_size: list,
        sigmax: int,
    ) -> np.array:
        """
        Convert the frame to grayscale and apply Gaussian blurring.

        Parameters
        ----------
        frame : np.array
            frame array read from the video capture
        kernel_size : list
            kernel size for GaussianBlur
        sigmax : int
            Standard deviation in the X direction of the Gaussian kernel

        Returns
        -------
        gray_frame : np.array
            grayscaled input frame
        blurred_frame : np.array
            Gaussian-blurred grayscaled input frame
        """
        # convert the frame to grayscale frame
        gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        # apply Gaussian blurring
        blurred_frame = cv2.GaussianBlur(gray_frame, kernel_size, sigmax)

        return gray_frame, blurred_frame

    def compute_background_subtracted_frame(
        self,
        blurred_frame,
        mean_blurred_frame,
        max_abs_blurred_frame,
    ):
        """
        Compute the background subtracted frame for the
        input blurred frame, given the mean and max absolute frames of
        its corresponding video.

        Parameters
        ----------
        blurred_frame : np.array
            Gaussian-blurred grayscaled input frame
        mean_blurred_frame : np.array
            mean of all blurred frames in the video
        max_abs_blurred_frame : np.array
            pixelwise max absolute value across all blurred frames in the video

        Returns
        -------
        background_subtracted_frame : np.array
            normalised difference between the blurred frame f and
            the mean blurred frame
        """
        return (
            ((blurred_frame - mean_blurred_frame) / max_abs_blurred_frame) + 1
        ) / 2

    def compute_motion_frame(
        self,
        frame_delta,
        background_subtracted_frame,
        mean_blurred_frame,
        max_abs_blurred_frame,
    ):
        """
        _summary_.

        Parameters
        ----------
        frame_delta : int
            difference in number of frames used to compute the motion
            channel is computed
        background_subtracted_frame : np.array
            normalised difference between the blurred frame f and
            the mean blurred frame
        mean_blurred_frame : np.array
            mean of all blurred frames in the video
        max_abs_blurred_frame : np.array
            pixelwise max absolute value across all blurred frames in the video

        Returns
        -------
        motion_frame : np.array
            absolute difference between the background subtracted frame f
            and the background subtracted frame f+delta
        """
        # compute the blurred frame frame_idx+delta
        _, blurred_frame_delta = self.apply_grayscale_and_blur(
            frame_delta,
            [5, 5],
            0,
        )
        # compute the background subtracted for frame_idx + delta
        background_subtracted_frame_delta = (
            self.compute_background_subtracted_frame(
                blurred_frame_delta,
                mean_blurred_frame,
                max_abs_blurred_frame,
            )
        )

        # compute the motion channel for frame_idx
        return np.abs(
            background_subtracted_frame_delta - background_subtracted_frame,
        )

    def __inference(
        self, final_frame: np.ndarray, frame: np.ndarray
    ) -> np.ndarray:
        """
        Perform inference on a single frame of the video.

        Args:
            frame (np.ndarray): The input frame as a NumPy array.

        Returns:
            None
        """
        self.trained_model.eval()
        transform = transforms.Compose(
            [
                transforms.ToTensor(),
            ]
        )
        img = transform(final_frame)
        img = img.to(device)

        img = img.unsqueeze(0)
        prediction = self.trained_model(img)
        pred_score = list(prediction[0]["scores"].detach().cpu().numpy())

        if not self.args.sort:
            from _inference import inference_detection

            frame_out = inference_detection(
                frame, prediction, pred_score, self.score_threshold
            )

        else:
            from _inference import inference_tracking

            frame_out = inference_tracking(
                frame,
                prediction,
                pred_score,
                self.score_threshold,
                self.sort_crab,
            )
        return frame_out

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

        if args.image_stack:
            frame_count = 0

            # get image size
            width = video.get(cv2.CAP_PROP_FRAME_WIDTH)
            height = video.get(cv2.CAP_PROP_FRAME_HEIGHT)

            # initialise array for mean blurred frame
            mean_blurred_frame = np.zeros((int(height), int(width)))

            # initialise array for max_abs_blurred_frame
            max_abs_blurred_frame = np.zeros((int(height), int(width)))

            while video.isOpened() and frame_count <= 500:
                ret, frame = video.read()
                if not ret:
                    print("No frame read. Exiting...")
                    break

                else:
                    # Apply transformations to the frame
                    gray_frame, blurred_frame = self.apply_grayscale_and_blur(
                        frame, [5, 5], 0
                    )

                    # accumulate blurred frames
                    mean_blurred_frame += blurred_frame

                    # accumulate max absolute values
                    max_abs_blurred_frame = np.maximum(
                        max_abs_blurred_frame, abs(blurred_frame)
                    )

                    frame_count += 1
            # compute the mean
            mean_blurred_frame = mean_blurred_frame / frame_count
            print(mean_blurred_frame.shape)
            # video.release()

        while video.isOpened():
            ret, frame = video.read()
            if not ret:
                print("No frame read. Exiting...")
                break

            if args.image_stack:
                gray_frame, blurred_frame = self.apply_grayscale_and_blur(
                    frame, [5, 5], 0
                )

                background_subtracted_frame = (
                    self.compute_background_subtracted_frame(
                        blurred_frame,
                        mean_blurred_frame,
                        max_abs_blurred_frame,
                    )
                )

                # Compute motion channel using the updated mean and max_abs values
                video.set(
                    cv2.CAP_PROP_POS_FRAMES,
                    video.get(cv2.CAP_PROP_POS_FRAMES) + 10,
                )
                success_delta, frame_delta = video.read()
                if not success_delta:
                    print("Cannot read frame. Exiting...")
                    break

                motion_frame = self.compute_motion_frame(
                    frame_delta,
                    background_subtracted_frame,
                    mean_blurred_frame,
                    max_abs_blurred_frame,
                )

                # Stack the channels
                final_frame = np.dstack(
                    [gray_frame, background_subtracted_frame, motion_frame]
                ).astype(np.float32)
                final_frame = (final_frame * 255).astype(np.uint8)

            else:
                final_frame = frame

            frame_out = self.__inference(final_frame, frame)
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
        default=False,
        help="running sort as tracker",
    )
    parser.add_argument(
        "--image_stack",
        type=bool,
        default=False,
        help="using additional channels images as the input",
    )

    args = parser.parse_args()
    inference = Detector_Inference(args)
    inference.inference_model()
