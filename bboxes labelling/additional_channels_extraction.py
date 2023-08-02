import os
import argparse
import cv2
import numpy as np
from PIL import Image
from pathlib import Path

from utils import read_json_file


def compute_stacked_inputs(args) -> None:
    """
    Function to compute grayscale, background subtracted and motion signal frame based

    Args:
        args (argparse.Namespace): An object containing
        the parsed command-line arguments.

    Returns:
        None

    References:
        https://github.com/visipedia/caltech-fish-counting

    """

    frame_dict = read_json_file(args.json_path)

    # Set batch size (number of frames per batch)
    batch_size = 1000

    for vid_file, list_frame_indices in frame_dict.items():
        if not os.path.exists(vid_file):
            print(f"Video path not found: {vid_file}")
            print(f"Skipped video {vid_file}")
            continue

        cap = cv2.VideoCapture(vid_file)
        n_frame = 0

        while True:
            frame_data = []
            for _ in range(batch_size):
                ret, frame = cap.read()

                if not ret:
                    # Break the loop if no more frames to read
                    break

                frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                frame_data.append(frame)
                n_frame += 1

            if not frame_data:
                break

            frames = np.stack(frame_data)

            # # Gaussian blurring
            blurred_frames = frames.astype(np.float32)

            for i in range(frames.shape[0]):
                blurred_frames[i] = cv2.GaussianBlur(
                    blurred_frames[i], args.kernel_size, args.sigmax
                )

            # average of all the frames after blurring
            # mean subtraction -- remove the overall brightness and
            # contrast differences caused by variations in the original frames
            # normalised the frame
            blurred_frames_mean = blurred_frames.mean(axis=0)
            norm_factor = np.max(np.abs(blurred_frames))
            background_subtraction = (
                (blurred_frames - blurred_frames_mean) / norm_factor + 1
            ) / 2

            # detecting motion by finding the differences between frame
            # set the delta : frame[i+delta] - frame[i]
            for i, frame_offset in enumerate(range(len(frames) - args.delta)):
                if (i + (n_frame - batch_size)) in list_frame_indices:
                    file_name = (
                        f"{Path(vid_file).parent.stem}_"
                        f"{Path(vid_file).stem}_"
                        f"frame_{i+(n_frame-batch_size):06d}.png"
                    )
                    frame_image = np.dstack(
                        [
                            frames[i] / 255,  # grayscale original frame
                            background_subtraction[i],  # foreground mask
                            np.abs(
                                background_subtraction[i + args.delta]
                                - background_subtraction[i]
                            ),  # motion mask
                        ]
                    ).astype(np.float32)
                    frame_image = (frame_image * 255).astype(np.uint8)
                    out_fp = os.path.join(args.out_dir, file_name)
                    Image.fromarray(frame_image).save(out_fp, quality=95)
            del frames
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
