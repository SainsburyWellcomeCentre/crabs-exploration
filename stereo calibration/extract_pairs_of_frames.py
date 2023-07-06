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
    - 'format', holds container-level info (i.e., info that applies to all streams)
    - 'streams', holding a list of dicts, one per stream

    Frame rate metrics:
    - r_frame_rate: the lowest common multiple of all the frame rates in the stream
    - avg_frame_rate: total # frames / total duration
    https://video.stackexchange.com/questions/20789/ffmpeg-default-output-frame-rate?newreg=e797b27b58a241dc9af8734dc8e14dc4

    The container has 3 streams:
    - codec_type: audio
      no frame rate,
      nb_frames = number of frames in **audio** stream (ok?)
    - codec_type: video
      frame rate as a fraction,
      nb_frames = total number of frames
      (extracted from metadata, not computed by ffmpeg directly decoding every frame)
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
        a dictionary with an entry for each video file that maps to a dictionary
        with the following keys:
            - r_frame_rate_str: ffprobe's r_frame_rate, expressed as a string fraction
            - n_frames: total number of frames
            - timecode_object: an instance of the Timecode class, using the above parameters

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
        tc_video = Timecode(r_frame_rate_str, start_timecode)

        # store data in dict
        timecodes_dict[video_path] = {
            "r_frame_rate_str": r_frame_rate_str,  # --- do I need this?
            "n_frames": n_frames,  # --- do I need this? yes, for checking with opencv later?
            # "start_timecode": start_timecode,
            "timecode_object": tc_video,
        }

    return timecodes_dict


def compute_synching_timecode(timecodes_dict: dict):
    """Determine the timecode for the syncing frame,
    aka the first frame whose timecode exists in all videos.

    We assume all videos in timecodes_dict were timecode-synched
    (i.e., their timecode streams will overlap from the syncing point
    onwards)

    Parameters
    ----------
    timecodes_dict : dict
        _description_

    Returns
    -------
    dict
        a dictionary with an entry for each video file that maps to a dictionary
        with the following keys:
            - r_frame_rate_str: ffprobe's r_frame_rate, expressed as a string fraction
            - n_frames: total number of frames
            - timecode_object: an instance of the Timecode class, using the above parameters
    """
    # find the video with the max start_timecode (syncing point)
    max_start_timecode = max(
        [vid["timecode_object"] for vid in timecodes_dict.values()]
    )

    # for vid in timecodes_dict.values():
    #     if vid["timecode_object"] == max_start_timecode:
    #         vid_max_timecode = vid
    #         break

    # # format as Timecode object
    # max_start_timecode = Timecode(
    #     vid_max_timecode["r_frame_rate_str"], max_start_timecode
    # )

    return max_start_timecode


def compute_opencv_start_idx(timecodes_dict, max_start_timecode):
    # compute difference between the starting timecode of each
    # video and the max (syncing) timecode
    # (that would be the index to start the capture in opencv,
    # because opencv uses 0-indexing for frames)
    for vid in timecodes_dict.values():
        if max_start_timecode == vid["timecode_object"]:
            # timecode does not accept subtraction of the same timecode
            vid["opencv_start_idx"] = 0  
        else:
            # I find this syntax a bit odd...
            # (why is this a timecode method but needs a timecode as an input?)
            vid["opencv_start_idx"] = max_start_timecode.tc_to_frames( 
                max_start_timecode - vid["timecode_object"]
            )

    return timecodes_dict


def extract_frames_from_video(
    video_path_str,
    nframes,
    opencv_start_idx,
    output_parent_dir: str = "./calibration_pairs",
):
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

    # extract frames
    # 'nframes' not included in loop because last index 0-based is nframes-1!
    pair_count = 0  # for consistency, pair_count is also 0-based
    for frame_idx0 in range(opencv_start_idx, opencv_start_idx+3): #nframes):

        # read frame
        success, frame = cap.read()

        # write frame to file
        if success:
            # filepath
            # 'cv' before frame number to hint that the indexing is 
            # opencv one, so 0-based?
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

    # Input data ----> change to argparse
    # videos_parent_dir = Path(
    #     "/Users/sofia/Documents_local/project_Zoo_crabs/crabs-exploration/crab_courtyard/"
    # )
    # file_types = "**/*.MOV"  # , "**/*.mp4", "**/*.avi")

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
    file_types = tuple(
        f"**/*.{ext}"
        for ext in args.video_extensions
    )

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

    # Compute syncing timecode: the max start timecode across all videos
    sync_timecode = compute_synching_timecode(timecodes_dict)

    # Compute opencv start index per video:
    timecodes_dict = compute_opencv_start_idx(timecodes_dict, sync_timecode)

    # Extract frames with opencv and save to directory
    for vid_str, vid_dict in timecodes_dict.items():
        extract_frames_from_video(
            vid_str,
            vid_dict["n_frames"],
            vid_dict["opencv_start_idx"],
            output_parent_dir=args.output_calibration_dir,
        )
