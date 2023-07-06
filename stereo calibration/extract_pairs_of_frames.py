import argparse
import logging
from pathlib import Path

import cv2
import ffmpeg
from timecode import Timecode


def compute_timecode_params_per_video(list_paths: list[Path]):
    """Compute timecode parameters per video

    We assume the timecode data is logged in the timecode stream,
    since we are expecting MOV files.

    On file types and timecode info (from ffprobe docs):
    - MPEG1/2 timecode is extracted from the GOP, and is available in the video
      stream details (-show_streams, see timecode).
    - MOV timecode is extracted from tmcd track, so is available in the tmcd
      stream metadata (-show_streams, see TAG:timecode).
    - DV, GXF and AVI timecodes are available in format metadata
      (-show_format, see TAG:timecode).

    FFprobe output is a (json) dict w/ two fields:
    - 'format', holds container-level info (i.e., info that applies to all 
       streams)
    - 'streams', holding a list of dicts, one per stream

    Frame rate metrics:
    - r_frame_rate: the lowest common multiple of all the frame rates in the 
      stream
    - avg_frame_rate: total # frames / total duration
    https://video.stackexchange.com/questions/20789/ffmpeg-default-output-frame-rate?newreg=e797b27b58a241dc9af8734dc8e14dc4

    The container has 3 streams:
    - codec_type: audio
      no frame rate,
      nb_frames = number of frames in **audio** stream (ok?)
    - codec_type: video
      frame rate as a fraction,
      nb_frames = total number of frames
      (extracted from metadata, not computed by ffmpeg directly decoding every 
       frame)
    - codec_type: 'data'
      'codec_tag_string': 'tmcd', nb_frames = 1
       the timecode stream also contains r_frame_rate and avg_frame_rate

    Parameters
    ----------
    list_paths : list[Path]
       list of Paths to video files to extract timecode from

    Returns
    -------
    dict
        a dictionary with an entry for each video file that maps to a 
        dictionary with the following keys:
            - r_frame_rate_str: ffprobe's r_frame_rate, expressed as a string 
              fraction
            - n_frames: total number of frames
            - start_timecode: an instance of the Timecode class, for the
              timecode of the first frame and at the frame rate extracted for
              the video
            - end_timecode: an instance of the Timecode class, for the
              timecode of the last frame and at the frame rate extracted for
              the video

    """
    timecodes_dict = {}
    for vid in list_paths:

        # run ffprobe on video
        video_path = str(vid)
        ffprobe_json = ffmpeg.probe(video_path)

        # parse streams
        # we assume one video stream only and one timecode stream
        for s in ffprobe_json["streams"]:
            if s["codec_type"] == "video":
                video_stream = s
            if s["codec_tag_string"] == "tmcd":
                tmcd_stream = s

        # extract data from video stream
        r_frame_rate_str = video_stream["r_frame_rate"]
        n_frames = int(video_stream["nb_frames"])

        # extract data from timecode stream
        start_timecode = tmcd_stream["tags"]["timecode"]

        # check timecode from tmcd stream matches timecode from format
        if ffprobe_json["format"]["tags"]["timecode"] != start_timecode:
            logging.error(
                "The start timecodes from the container format"
                "and the video stream don't match"
            )
            break

        # check tmcd avg_frame_rate matches r_frame_rate from video
        if tmcd_stream["avg_frame_rate"] != r_frame_rate_str:
            logging.error(
                "ERROR: the frame rates from the timecode"
                " and video stream don't match"
            )
            break

        # instantiate a timecode object for this video
        # (linked to the start timecode)
        start_timecode = Timecode(r_frame_rate_str, start_timecode)

        # compute end timecode
        end_timecode_tuple = start_timecode.frames_to_tc(
            start_timecode.frames + n_frames - 1  
            # do not count the first frame twice!
        )
        end_timecode_str = start_timecode.tc_to_string(*end_timecode_tuple)
        end_timecode = Timecode(r_frame_rate_str, end_timecode_str)

        # store data in dict
        timecodes_dict[video_path] = {
            "r_frame_rate_str": r_frame_rate_str,
            "n_frames": n_frames, 
            "start_timecode": start_timecode,
            "end_timecode": end_timecode,
        }

    return timecodes_dict


def compute_synching_timecodes(timecodes_dict: dict) -> tuple[Timecode]:
    """Determine the timecodes for the first and last frame
    in common across all videos

    We assume all videos in timecodes_dict were timecode-synched
    (i.e., their timecode streams will overlap from the common start frame
    aka syncing point until the common end frame)

    Parameters
    ----------
    dict
        a dictionary with an entry for each video file that maps to a 
        dictionary with the following keys:
            - r_frame_rate_str: ffprobe's r_frame_rate, expressed as a string 
              fraction
            - n_frames: total number of frames
            - start_timecode: an instance of the Timecode class, for the
              timecode of the first frame and at the frame rate extracted for
              the video
            - end_timecode: an instance of the Timecode class, for the
              timecode of the last frame and at the frame rate extracted for
              the video

    Returns
    -------
    tuple of Timecodes
        a tuple containing the max and min timecodes in common across all
        videos

    """
    # compute the max start_timecode (syncing point)
    max_start_timecode = max(
        [
            vid["start_timecode"] 
            for vid in timecodes_dict.values()
        ]
    )

    # compute the min end_timecode
    min_end_timecode = min(
        [
            vid["end_timecode"] 
            for vid in timecodes_dict.values()
        ]
    )

    return (max_start_timecode, min_end_timecode)


def compute_opencv_start_idx(
    timecodes_dict, 
    max_start_timecode, 
    min_end_timecode
):
    """
    Compute the start and end indices for opencv
    based on the common starting frame (max_start_timecode)
    and the common end frame (min_end_timecode).

    This function adds the following fields to the input dict:
    - open_cv_start_idx
    - open_cv_end_idx
    Both are 0-based indices relative to the start of the video
    (i.e., the first frame of the video has open_cv_idx = 0)

    Note:
    - timecode does not accept subtraction of the same timecode
      (a timecode cannot have 0 frames)

    Parameters
    ----------
    timecodes_dict : dict
        _description_
    max_start_timecode : Timecode
        _description_
    min_end_timecode : Timecode
        _description_

    Returns
    -------
    dict
        _description_
    """

    for vid in timecodes_dict.values():
        # Set opencv start idx from the video's start_timecode
        if max_start_timecode == vid["start_timecode"]:

            vid["opencv_start_idx"] = 0
        else:
            vid["opencv_start_idx"] = (
                max_start_timecode - vid["start_timecode"]
            ).frames
            # vid["opencv_start_idx"] = max_start_timecode.tc_to_frames(
            #     max_start_timecode - vid["start_timecode"]
            # )

        # Set opencv end_idx from the video's end_timecode
        if min_end_timecode == vid["end_timecode"]:
            vid["opencv_end_idx"] = vid["n_frames"] - 1  # 0-based indexing
        else:
            vid["opencv_end_idx"] = (
                min_end_timecode - vid["start_timecode"]
            ).frames

    return timecodes_dict


def extract_frames_from_video(
    video_path_str,
    nframes,
    opencv_start_idx,
    opencv_end_idx,
    output_parent_dir: str = "./calibration_pairs",
):
    """Extract frames between selected indices from video,
    and save them to the output directory

    Parameters
    ----------
    video_path_str : _type_
        _description_
    nframes : _type_
        _description_
    opencv_start_idx : _type_
        _description_
    opencv_end_idx : _type_
        _description_
    output_parent_dir : str, optional
        _description_, by default "./calibration_pairs"
    """

    # initialise capture
    cap = cv2.VideoCapture(video_path_str)
    if cap.get(cv2.CAP_PROP_FRAME_COUNT) != nframes:
        logging.error(
            "The total number of frames from ffmpeg and opencv don't match"
        )

    # set capture index to desired starting point
    # ATT! Opencv is 0-based indexed (aka first frame is index 0)
    cap.set(cv2.CAP_PROP_POS_FRAMES, opencv_start_idx)

    # create output dir for this video
    output_dir_one_camera = Path(output_parent_dir) / Path(video_path_str).stem
    output_dir_one_camera.mkdir(parents=True, exist_ok=True)

    # extract frames between start index and end index
    pair_count = 0  # for consistency, pair_count is also 0-based
    for frame_idx0 in range(opencv_start_idx, opencv_end_idx + 1):

        # read frame
        success, frame = cap.read()

        # write frame to file
        if success:
            # filepath
            file_path = (
                output_dir_one_camera
                / f"frame{frame_idx0:05d}_pair{pair_count:03d}.png"
            )

            # write to file
            flag_saved = cv2.imwrite(str(file_path), frame)

            # check if saved correctly
            if flag_saved:
                logging.info(f"frame {frame_idx0} saved at {file_path}")
            else:
                logging.warning(
                    f"ERROR saving {Path(video_path_str).stem}, "
                    f"frame {frame_idx0}...skipping"
                )
                continue

            # increase pair count
            pair_count += 1


# ----------
# Main
# ------------
if __name__ == "__main__":

    # Input data
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "input_videos_parent_dir",  # positional, needs first fwd slash!
        help="path to directory with videos",
    )
    parser.add_argument(
        "--video_extensions",
        nargs="*",
        default=["MOV"],
        help="video extensions to consider (typically MOV, mp4, avi)",
    )
    parser.add_argument(
        "--output_calibration_dir",
        default="./calibration_pairs",  # does this work?
        help=(
            "path to directory in which to store extracted"
            " frames (by default, the current directory)"
        ),
    )
    args = parser.parse_args()

    # Transform file_types
    file_types = tuple(f"**/*.{ext}" for ext in args.video_extensions)

    # Extract list of files
    list_paths = []
    for typ in file_types:
        list_paths.extend(
            [
                p
                for p in list(Path(args.input_videos_parent_dir).glob(typ))
                if not p.name.startswith("._")
            ]
        )

    # Extract timecode params
    timecodes_dict = compute_timecode_params_per_video(list_paths)

    # Compute syncing timecodes: the max start timecode across all videos
    # and the min end timecode
    max_start_timecode, min_end_timecode = compute_synching_timecodes(timecodes_dict)


    # Compute opencv start and end indices per video:
    timecodes_dict = compute_opencv_start_idx(
        timecodes_dict, 
        max_start_timecode, 
        min_end_timecode
    )

    # Extract synced frames with opencv and save to directory
    for vid_str, vid_dict in timecodes_dict.items():
        extract_frames_from_video(
            vid_str,
            vid_dict["n_frames"],
            vid_dict["opencv_start_idx"],
            vid_dict["opencv_end_idx"],
            output_parent_dir=args.output_calibration_dir,
        )
