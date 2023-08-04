import os
import argparse
import cv2
import numpy as np
from PIL import Image
from pathlib import Path

from utils import read_json_file


def apply_transform(frame: np.array, kernel_size: list, sigmax: int) -> np.array:
    """Function to apply transformation to the frame.
    Convert the frame to grayscale and apply Gaussian blurring

    Args:
        frame (np.array): frame array read from the video capture
        kernel_size (list): kernel size for GaussianBlur
        sigmax (_type_): Standard deviation in the X direction of the Gaussian kernel

    Returns:
        np.array: converted frame to grayscale and blurred frame
    """

    # convert the frame to grayscale frame
    gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    # apply Gaussian blurring
    blurred_frame = cv2.GaussianBlur(gray_frame, kernel_size, sigmax)

    return gray_frame, blurred_frame


def compute_stacked_inputs(args: argparse.Namespace) -> None:
    """
    Function to compute grayscale, background subtracted and motion signal frame based

    Args:
        args (argparse.Namespace): An object containing
        the parsed command-line arguments.

    References:
        https://github.com/visipedia/caltech-fish-counting

    """

    # get video files and their frame indices
    frame_dict = read_json_file(args.json_path)

    for vid_file, list_frame_indices in frame_dict.items():
        if not os.path.exists(vid_file):
            print(f"Video path not found: {vid_file}. Skip video")
            continue

        print(vid_file)
        # Initialise video capture
        cap = cv2.VideoCapture(vid_file)

        frame_counter = 0

        # get image size
        width = cap.get(cv2.CAP_PROP_FRAME_WIDTH)
        height = cap.get(cv2.CAP_PROP_FRAME_HEIGHT)

        # initialise array for mean blurred frame
        mean_blurred_frame = np.zeros((int(height), int(width)))

        # initialise array for max_abs_blurred_frame
        max_abs_blurred_frame = np.zeros((int(height), int(width)))

        while True:
            ret, frame = cap.read()

            if not ret:
                # Break the loop if no more frames to read
                print(f"Cannot read the frame{frame_counter}. Exiting...")
                break

            # apply transformations to the frame
            _, blurred_frame = apply_transform(frame, args.kernel_size, args.sigmax)
            # accumulate blurred frames
            mean_blurred_frame += blurred_frame

            # accumulate max absolute values
            max_abs_blurred_frame = max_abs_blurred_frame = np.maximum(
                max_abs_blurred_frame, abs(blurred_frame)
            )

            # update frame counter
            frame_counter += 1

        # compute the mean
        mean_blurred_frame = mean_blurred_frame / frame_counter
        # save the mean
        cv2.imwrite(f"{Path(vid_file).stem}_mean.jpg", mean_blurred_frame)

        # for every frame extracted for labelling
        for frame_idx in list_frame_indices:
            # read the frame from the video
            cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
            ret, frame = cap.read()

            if not ret:
                # Break the loop if no more frames to read
                print(f"Cannot read frame{frame_idx}. Exiting...")
                break

            # apply transformations to the frame
            gray_frame, blurred_frame = apply_transform(
                frame, args.kernel_size, args.sigmax
            )

            # compute the background subtracted frame
            background_subtracted_frame = (
                ((blurred_frame - mean_blurred_frame) / max_abs_blurred_frame) + 1
            ) / 2

            # compute the motion frame
            # read frame f-delta, the frame delta frames before the current one
            # TODO: how to deal with frame_idx - delta < 0.
            cap.set(cv2.CAP_PROP_POS_FRAMES, (frame_idx + args.delta))
            success_delta, frame_delta = cap.read()

            if not success_delta:
                # Break the loop if no more frames to read
                print(f"Cannot read frame{frame_idx}+{args.delta}. Exiting...")
                break

            _, blurred_frame_delta = apply_transform(
                frame_delta, args.kernel_size, args.sigmax
            )
            # compute the background subtracted for frame frame_idx + delta
            background_subtracted_frame_delta = (
                ((blurred_frame_delta - mean_blurred_frame) / max_abs_blurred_frame) + 1
            ) / 2

            # compute the motion channel for frame frame_idx
            motion_frame = np.abs(
                background_subtracted_frame_delta - background_subtracted_frame
            )

            # stack the three channels
            final_frame = np.dstack(
                [
                    gray_frame,  # original grayscaled image
                    background_subtracted_frame,  # background-subtracted
                    motion_frame,  # motion signal
                ]
            ).astype(np.float32)
            final_frame = (final_frame * 255).astype(np.uint8)

            # save final frame as file
            file_name = (
                f"{Path(vid_file).parent.stem}_"
                f"{Path(vid_file).stem}_"
                f"frame_{frame_idx:06d}.png"
            )
            out_fp = os.path.join(args.out_dir, file_name)
            Image.fromarray(final_frame).save(out_fp, quality=95)

        cap.release()


def argument_parser() -> argparse.Namespace:
    """
    Parse command-line arguments for your script.

    Returns:
        argparse.Namespace: An object containing the parsed command-line arguments.
                            The attributes of this object correspond to the defined
                            command-line arguments in your script.
    """

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--json_path",
        type=str,
        required=True,
        help="Location of json file with list frame indices.",
    )
    parser.add_argument(
        "--out_dir",
        type=str,
        required=True,
        help="Output location for converted frames.",
    )
    parser.add_argument(
        "--kernel_size",
        nargs=2,
        type=int,
        default=[5, 5],
        help="Kernel size for the Gaussian blur (default: 5 5)",
    )
    parser.add_argument(
        "--sigmax",
        type=int,
        default=0,
        help="Standard deviation in the X direction of the Gaussian kernel",
    )
    parser.add_argument(
        "--delta",
        type=int,
        default=10,
        help="The value how many frame differences we compute",
    )
    args = parser.parse_args()
    return args


if __name__ == "__main__":
    args = argument_parser()

    compute_stacked_inputs(args)
