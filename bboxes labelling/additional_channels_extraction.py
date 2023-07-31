# based on https://github.com/visipedia/caltech-fish-counting

import os
import argparse
import json
import cv2
import numpy as np
from PIL import Image
from pathlib import Path


def read_json_file(file_path):
    try:
        with open(file_path, "r") as file:
            data = json.load(file)
        return data
    except FileNotFoundError:
        print(f"File not found: {file_path}")
        return None
    except json.JSONDecodeError:
        print(f"Error decoding JSON data from file: {file_path}")
        return None


def get_frames(args):
    frame_dict = read_json_file(args.json_path)

    # Set batch size (number of frames per batch)
    batch_size = 1000

    for vid_file, frame_idx in frame_dict.items():
        if not os.path.exists(vid_file):
            print(f"Video path not found: {vid_file}")
            continue

        cap = cv2.VideoCapture(vid_file)
        frame_data = []
        n_frame = 0

        while True:
            frames = []
            for _ in range(batch_size):
                ret, frame = cap.read()

                if not ret:
                    # Break the loop if no more frames to read
                    break

                frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                # Gaussian blurring
                blurred_frames = cv2.GaussianBlur(frame, (5, 5), 0)
                frames.append(blurred_frames)
                n_frame += 1

            if not frames:
                break
            blurred_frames = np.stack(frames).astype(np.float32)
            # print(blurred_frames.shape)

            # average of all the frames after blurring
            mean_blurred_frame = blurred_frames.mean(axis=0)
            # mean subtraction -- remove the overall brightness and contrast differences caused by variations in the original frames
            blurred_frames -= mean_blurred_frame
            # normalised the frame
            mean_normalization_value = np.max(np.abs(blurred_frames))
            blurred_frames /= mean_normalization_value
            blurred_frames += 1
            blurred_frames /= 2

            delta = 1

            # detecting motion by finding the differences between frame
            # set the delta : frame[i+delta] - frame[i]
            for i, frame_offset in enumerate(range(len(frames) - delta)):
                if (i + (n_frame - batch_size)) in frame_idx:
                    file_name = (
                        f"{Path(vid_file).parent.stem}_"
                        f"{Path(vid_file).stem}_"
                        f"frame_{i+(n_frame-batch_size):06d}.png"
                    )
                    frame_image = np.dstack(
                        [
                            frames[i] / 255,  # grayscale original frame
                            blurred_frames[i],  # foreground mask
                            np.abs(
                                blurred_frames[i + delta] - blurred_frames[i]
                            ),  # motion mask
                        ]
                    ).astype(np.float32)
                    frame_image = (frame_image * 255).astype(np.uint8)
                    out_fp = os.path.join(args.out_dir, file_name)
                    Image.fromarray(frame_image).save(out_fp, quality=95)

            del frames

        cap.release()


def argument_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--json_path",
        default="Data/train_data/extracted_frames.json",
        help="Location of json file with frame_idx.",
    )
    parser.add_argument(
        "--out_dir",
        default="Data/train_data/bg_sub/",
        help="Output location for converted frames.",
    )
    return parser


if __name__ == "__main__":
    args = argument_parser().parse_args()

    get_frames(args)
