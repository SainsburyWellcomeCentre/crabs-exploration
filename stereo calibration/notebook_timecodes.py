# %%
# - timecode
# - ffmpeg python bindings
from pathlib import Path
import cv2
import ffmpeg
from timecode import Timecode
import logging

from extract_pairs_of_frames import *

# %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
# Input data
input_videos_parent_dir = "/Users/sofia/Documents_local/project_Zoo_crabs/crabs-exploration/crab_courtyard/"
video_extensions = ["MOV"]
output_calibration_dir = "./calibration_pairs"

# Transform file_types
file_types = tuple(
    f"**/*.{ext}"
    for ext in video_extensions
)

# Extract list of files
list_paths = []
for typ in file_types:
    list_paths.extend(
        [
            p
            for p in list(Path(input_videos_parent_dir).glob(typ))
            if not p.name.startswith("._")
        ]
    )


# %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
# Extract timecode params
timecodes_dict = compute_timecode_params_per_video(list_paths)


# %%%%%%%%%%%%%%%%%%%%%%%%%%%
# Compute syncing timecode: the max start timecode across all videos
sync_timecode = compute_synching_timecode(timecodes_dict)

# %%
# Compute opencv start index per video:
timecodes_dict = compute_opencv_start_idx(timecodes_dict, sync_timecode)

# %%
# Extract frames with opencv and save to directory
for vid_str, vid_dict in timecodes_dict.items():
    extract_frames_from_video(
        vid_str,
        vid_dict["n_frames"],
        vid_dict["opencv_start_idx"],
        output_parent_dir=output_calibration_dir,
    )


# %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
# Print timecodes for every frame in a sample video
# ATT!
# - Setting the framerate will automatically set the :attr:`.drop_frame`
#   attribute to correct value.
# - "Frame rates 29.97 and 59.94 are always drop frame"
#   ---> is this a standard thing? Kinda, in the sense that they are non-integer frame rates
# - I can change the default behaviour w/ force_non_drop_frame (Could be useful? Not for now tho)
# - for 59.94: 4 frames dropped when turning to the next minute! 
#   http://www.davidheidelberger.com/2010/06/10/drop-frame-timecode/ 
video_path = str(list_paths[1])
n_frames = timecodes_dict[video_path]["n_frames"]

tc_video_1 = timecodes_dict[video_path][
    "timecode_object"
]  # Timecode(r_frame_rate_str, start_timecode)

tc_video_1.frames  # frames elapsed from timecode '23:59:59:<last integer frame from fps>'
# (so frame 1 corresponds to timecode '00:00:00:00') ---> so like timecode of first frame, in frames?
tc_video_1.frame_number  # 0-based frame number
tc_video_1.framerate  # as a string rational number
tc_video_1.drop_frame  # bool

# OJO frames are 1-indexed right?
for frames_to_add in range(n_frames):

    # add frames to initial one
    curr_frame_number = tc_video_1.frames + frames_to_add  # 1-based

    # compute timecode
    current_frame_timecode = tc_video_1.tc_to_string(
        *tc_video_1.frames_to_tc(curr_frame_number)
    )

    print(
        f"Frame {curr_frame_number - tc_video_1.frames + 1}"
        f" \t Timecode {current_frame_timecode}"
    )


# .frame_number property
# methods:
# - frames_to_tc
# - tc_to_frames

# Drop frame?


# %%
