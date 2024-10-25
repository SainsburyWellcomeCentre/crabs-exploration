"""Script to extract pairs of frames for stereo calibration."""

import logging
from pathlib import Path

import cv2
import ffmpeg  # type: ignore
import typer
from timecode import Timecode


def compute_timecode_params_per_video(list_paths: list[Path]) -> dict:
    """Compute timecode parameters per video.

    We assume the timecode data is logged in the timecode stream ("tmcd"),
    since we are expecting MOV files (see Notes for further details).

    TODO: the timecodes obtained with ffprobe and Quicktime are
    different, we need to find out why. See issue here:
    https://github.com/SainsburyWellcomeCentre/crabs-exploration/issues/90

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
        - n_frames: total number of frames derived from ffmpeg
        - start_timecode: an instance of the Timecode class, for the
            timecode of the first frame and at the frame rate extracted for
            the video
        - end_timecode: an instance of the Timecode class, for the
            timecode of the last frame and at the frame rate extracted for
            the video

    Notes
    -----
    On file types and timecode info (from ffprobe docs):
    - MPEG1/2 timecode is extracted from the GOP, and is available in the video
      stream details (-show_streams, see timecode).
    - MOV timecode is extracted from tmcd track, so is available in the tmcd
      stream metadata (-show_streams, see TAG:timecode).
    - DV, GXF and AVI timecodes are available in format metadata
      (-show_format, see TAG:timecode).

    FFprobe output is a (json) dict with two fields:
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


def compute_synching_timecodes(
    timecodes_dict: dict,
) -> tuple[Timecode, Timecode]:
    """Determine timecodes for first and last common frames across all videos.

    We assume all videos in timecodes_dict were timecode-synched
    (i.e., their timecode streams will overlap from the common start frame
    - aka syncing point - until the common end frame)

    TODO: the timecodes obtained with ffprobe and Quicktime are
    different, we need to find out why. See issue here:
    https://github.com/SainsburyWellcomeCentre/crabs-exploration/issues/90

    Parameters
    ----------
    timecodes_dict: dict
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
    tuple[Timecode, Timecode]
        a tuple containing the max and min timecodes in common across all
        videos

    """
    # compute the max start_timecode (syncing point)
    max_start_timecode = max(
        [vid["start_timecode"] for vid in timecodes_dict.values()]
    )

    # compute the min end_timecode
    min_end_timecode = min(
        [vid["end_timecode"] for vid in timecodes_dict.values()]
    )

    return (max_start_timecode, min_end_timecode)


def compute_opencv_start_idx(
    timecodes_dict: dict,
    max_min_timecode: tuple[Timecode, Timecode],
) -> dict:
    """Compute start and end indices of a set of videos.

    Compute start and end indices of a set of videos
    for opencv tools, based on their common starting frame
    (max_start_timecode) and their common end frame
    (min_end_timecode).

    This function adds the following fields to the input dict:
    - open_cv_start_idx
    - open_cv_end_idx
    Both are 0-based indices relative to the start of the video
    (i.e., the first frame of the video has open_cv_idx = 0)

    Note:
    - timecode does not accept subtraction of the same timecode
      (a timecode cannot have 0 frames)

    TODO: the timecodes obtained with ffprobe and Quicktime are
    different, we need to find out why. See issue here:
    https://github.com/SainsburyWellcomeCentre/crabs-exploration/issues/90

    Parameters
    ----------
    timecodes_dict : dict
        A dictionary with an entry for each video file that maps to a
        dictionary with the following keys:
        - r_frame_rate_str: ffprobe's r_frame_rate, expressed as a string
            fraction
        - n_frames: total number of frames derived from ffmpeg
        - start_timecode: an instance of the Timecode class, for the
            timecode of the first frame and at the frame rate extracted for
            the video
        - end_timecode: an instance of the Timecode class, for the
            timecode of the last frame and at the frame rate extracted for
            the video
    max_min_timecode : tuple[Timecode, Timecode]
        a tuple containing the max_start_timecode (i.e., the common starting
        frame), and the min_end_timecode (i.e. the common end frame)

    Returns
    -------
    dict
        an extension to the timecodes_dict, with the openCV start and
        end indices per video. Both are 0-based indices relative to the
        start of the video.

    """
    (max_start_timecode, min_end_timecode) = max_min_timecode

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


def extract_chessboard_frames_from_video(
    video_path_str: str,
    video_dict: dict,
    chessboard_config: dict,
    output_parent_dir: str = "./calibration_pairs",
):
    """Extract frames with a chessboard pattern between the selected indices.

    TODO: detecting the checkerboard is very slow with open-cv if no board is
    present. See issue here:
    https://github.com/SainsburyWellcomeCentre/crabs-exploration/issues/90

    Parameters
    ----------
    video_path_str : str
        path to the video to analyse
    video_dict : dict
        a dictionary holding timecode parameters per video.
        It should have at least the following keys:
        - n_frames: number of frames from ffmpeg
        - opencv_start_idx: start index for synced period
        - opencv_end_idx: end index for synced period
    chessboard_config : dict
        A dictionary specifying the number of rows and columns of the
        chessboard pattern.
    output_parent_dir : str, optional
        directory to which save the extracted synced frames,
        by default "./calibration_pairs"

    """
    # initialise capture
    cap = cv2.VideoCapture(video_path_str)
    if cap.get(cv2.CAP_PROP_FRAME_COUNT) != video_dict["n_frames"]:
        logging.error(
            "The total number of frames from ffmpeg and opencv don't match"
        )

    # set capture index to desired starting point
    # ATT! Opencv is 0-based indexed (aka first frame is index 0)
    cap.set(cv2.CAP_PROP_POS_FRAMES, video_dict["opencv_start_idx"])

    # create output dir for this video
    output_dir_one_camera = Path(output_parent_dir) / Path(video_path_str).stem
    output_dir_one_camera.mkdir(parents=True, exist_ok=True)

    # extract frames between start index and end index
    # if a chessboard pattern is detected
    pair_count = 0  # for consistency, pair_count is also 0-based
    for frame_idx0 in range(
        video_dict["opencv_start_idx"], video_dict["opencv_end_idx"] + 1
    ):
        # read frame
        success, frame = cap.read()

        # if frame is read successfully
        if success:
            # ---------------
            # Find the chessboard corners
            # If desired number of corners are found in the image then
            # ret = true
            # TODO: append 2d coords of corners?
            frame_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

            # TODO: the following is very slow when no chessboard is present
            ret, corners = cv2.findChessboardCorners(
                frame_gray,
                (chessboard_config["rows"], chessboard_config["cols"]),
                None,
            )
            # cv2.CALIB_CB_ADAPTIVE_THRESH + cv2.CALIB_CB_FAST_CHECK +
            # cv2.CALIB_CB_NORMALIZE_IMAGE
            # -------------
            if ret:
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

                # increase pair count -------> review this!
                pair_count += 1

            else:
                logging.warning(
                    "WARNING: No chessboard detected on"
                    f" {Path(video_path_str).stem}, "
                    f"frame {frame_idx0}...skipping"
                )


def main(
    input_videos_parent_dir: str,
    video_extensions: list,
    output_calibration_dir: str = "./calibration_pairs",
):
    """Extract pairs of frames for stereo calibration.

    Parameters
    ----------
    input_videos_parent_dir : str
        path to directory with videos
    video_extensions : list
        video extensions to consider (typically MOV, mp4, avi)
    output_calibration_dir : str, optional
        path to directory in which to store extracted frames,
        by default "./calibration_pairs"

    """
    # Transform extensions to file_types regular expressions
    file_types = tuple(f"**/*.{ext}" for ext in video_extensions)

    # Extract list of video files to process
    list_paths = []
    for typ in file_types:
        list_paths.extend(
            [
                p
                for p in list(Path(input_videos_parent_dir).glob(typ))
                if not p.name.startswith("._")
            ]
        )

    # Extract timecode parameters for each video
    timecodes_dict = compute_timecode_params_per_video(list_paths)

    # Compute syncing timecodes: the max start timecode across all videos
    # and the min end timecode
    max_start_timecode, min_end_timecode = compute_synching_timecodes(
        timecodes_dict
    )

    # Compute opencv start and end frame indices per video
    timecodes_dict = compute_opencv_start_idx(
        timecodes_dict,
        (max_start_timecode, min_end_timecode),
    )

    # Extract pairs of sync frames with a visible chessboard
    # and save to directory
    chessboard_config = {
        "rows": 6,  # ATT! THESE ARE INNER POINTS ONLY
        "cols": 9,  # ATT! THESE ARE INNER POINTS ONLY
    }
    for vid_str, vid_dict in timecodes_dict.items():
        extract_chessboard_frames_from_video(
            vid_str,
            vid_dict,
            chessboard_config,
            output_parent_dir=output_calibration_dir,
        )


if __name__ == "__main__":
    typer.run(main)
