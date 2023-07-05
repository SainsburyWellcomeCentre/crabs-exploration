import ffmpeg
from timecode import Timecode
from pathlib import Path


def compute_timecode_params_per_video(
        list_paths: list[Path]
):
    """Compute timecode parameters per video

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
            - start_timecode: timecode of the first frame in the video (wrt origin 00:00:00:00)
    """
    timecodes_dict = {}
    for vid in list_paths:

        # run ffprobe on video
        video_path = str(vid)
        ffprobe_json = ffmpeg.probe(video_path)

        # parse streams
        #  we assume one video stream only and one timecode stream
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

        # check timecode from tmcs matches timecode from format
        if ffprobe_json["format"]["tags"]["timecode"] != start_timecode:
            print("ERROR: ")
            break

        # check tmcd average frame rate matches r_frame rate from video
        if (tmcd_stream["avg_frame_rate"] != r_frame_rate_str):
            print(f"ERROR: timecode and video frame rates don't match")
            break

        # save data
        timecodes_dict[video_path] = {
            "r_frame_rate_str": r_frame_rate_str,
            "n_frames": n_frames,
            "start_timecode": start_timecode,
        }

    return timecodes_dict


if __name__ == '__main__':

    # Input data ----> change to argparse
    videos_parent_dir = Path(
        "/Users/sofia/Documents_local/project_Zoo_crabs/crabs-exploration/crab_courtyard/"
    )
    file_types = ("**/*.MOV", "**/*.mp4", "**/*.avi")
    list_paths = []
    for typ in file_types:
        list_paths.extend(
            [p for p in list(videos_parent_dir.glob(typ)) if not p.name.startswith("._")]
        )

    # Extract timecode params
    timecodes_dict = compute_timecode_params_per_video(list_paths)

    # Select the largest starting timecode ---> that is the syncing point
    timecode_sync = max(
        [
            vid['start_timecode'] 
            for vid in timecodes_dict
        ]
    )

    # find corresponding frame number in either video and extract with opencv