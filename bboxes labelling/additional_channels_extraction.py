import os
import argparse
import cv2
import numpy as np
from PIL import Image
from pathlib import Path

from utils import read_json_file


def apply_grayscale_and_blur(
    frame: np.array, kernel_size: list, sigmax: int
) -> np.array:
    """Convert the frame to grayscale and apply Gaussian blurring

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


def compute_mean_and_max_abs_blurred_frame(cap, kernel_size, sigmax):
    """Compute the mean blurred frame and the maximum absolute-value
    blurred frame for a video capture cap

    Parameters
    ----------
    cap : _type_
        OpenCV VideoCapture object
    kernel_size : list
        kernel size for GaussianBlur
    sigmax : int
        Standard deviation in the X direction of the Gaussian kernel

    Returns
    -------
    mean_blurred_frame : np.array
        mean of all blurred frames in the video
    max_abs_blurred_frame : np.array
        pixelwise max absolute value across all blurred frames in the video
    """
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
        _, blurred_frame = apply_grayscale_and_blur(frame, kernel_size, sigmax)

        # accumulate blurred frames
        mean_blurred_frame += blurred_frame

        # accumulate max absolute values
        max_abs_blurred_frame = np.maximum(max_abs_blurred_frame, abs(blurred_frame))

        # update frame counter
        frame_counter += 1

    # compute the mean
    mean_blurred_frame = mean_blurred_frame / frame_counter

    return mean_blurred_frame, max_abs_blurred_frame


def compute_background_subtracted_frame(
    blurred_frame, mean_blurred_frame, max_abs_blurred_frame
):
    """Compute the background subtracted frame for the
    input blurred frame, given the mean and max absolute frames of
    its corresponding video

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
    background_subtracted_frame = (
        ((blurred_frame - mean_blurred_frame) / max_abs_blurred_frame) + 1
    ) / 2

    return background_subtracted_frame


def compute_motion_frame(
    frame_delta, background_subtracted_frame, mean_blurred_frame, max_abs_blurred_frame
):
    """_summary_

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
    _, blurred_frame_delta = apply_grayscale_and_blur(
        frame_delta, args.kernel_size, args.sigmax
    )
    # compute the background subtracted for frame_idx + delta
    background_subtracted_frame_delta = compute_background_subtracted_frame(
        blurred_frame_delta, mean_blurred_frame, max_abs_blurred_frame
    )

    # compute the motion channel for frame_idx
    motion_frame = np.abs(
        background_subtracted_frame_delta - background_subtracted_frame
    )

    return motion_frame


def compute_stacked_inputs(args: argparse.Namespace) -> None:
    """Compute the stacked inputs consist of
    grayscale, background subtracted and motion signal

    Parameters
    ----------
    args : argparse.Namespace
        An object containing the parsed command-line arguments.

    Notes
    -----
        This implementation follows the one at
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

        # Compute mean and max frames for this video
        (
            mean_blurred_frame,
            max_abs_blurred_frame,
        ) = compute_mean_and_max_abs_blurred_frame(cap, args.kernel_size, args.sigmax)

        # save the mean
        cv2.imwrite(f"{Path(vid_file).stem}_mean.jpg", mean_blurred_frame)

        # Compute channels for every frame extracted for labelling
        for frame_idx in list_frame_indices:
            # read the frame f from the video
            cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
            ret, frame = cap.read()
            if not ret:
                # Break the loop if no more frames to read
                print(f"Cannot read frame{frame_idx}. Exiting...")
                break

            # apply transformations to the frame
            gray_frame, blurred_frame = apply_grayscale_and_blur(
                frame, args.kernel_size, args.sigmax
            )

            # compute the background subtracted frame
            background_subtracted_frame = compute_background_subtracted_frame(
                blurred_frame, mean_blurred_frame, max_abs_blurred_frame
            )

            # read frame f+delta, the frame delta after before the current one
            cap.set(cv2.CAP_PROP_POS_FRAMES, (frame_idx + args.delta))
            success_delta, frame_delta = cap.read()
            if not success_delta:
                # Break the loop if no more frames to read
                print(f"Cannot read frame{frame_idx}+{args.delta}. Exiting...")
                break

            # compute motion channel
            motion_frame = compute_motion_frame(
                frame_delta,
                background_subtracted_frame,
                mean_blurred_frame,
                max_abs_blurred_frame,
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
    """Parse command-line arguments for the script.

    Returns
    -------
    argparse.Namespace
        An object containing the parsed command-line arguments.
        The attributes of this object correspond to the defined
        command-line arguments in the script.
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
        default=100,
        help="The value how many frame differences we compute",
    )
    args = parser.parse_args()
    return args


if __name__ == "__main__":
    args = argument_parser()

    compute_stacked_inputs(args)
