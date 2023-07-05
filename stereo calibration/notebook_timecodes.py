# %%
# - timecode
# - ffmpeg python bindings
import ffmpeg
from timecode import Timecode
import pathlib as pl

# %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
# Input data
videos_parent_dir = pl.Path(
    "/Users/sofia/Documents_local/project_Zoo_crabs/crabs-exploration/crab_courtyard/"
)
file_types = ("**/*.MOV", "**/*.mp4", "**/*.avi")
list_paths = []
for typ in file_types:
    list_paths.extend(
        [p for p in list(videos_parent_dir.glob(typ)) if not p.name.startswith("._")]
    )


# %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
# Get timecode data for each file (using ffmpeg)

def compute_timecode_vars_per_video(list_paths: list[pl.Path]):
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

# execute
timecodes_dict = compute_timecode_vars_per_video(list_paths)

# NOTES:
# FFprobe output is a (json) dict w/ two fields:
# - 'format', holds container-level info (i.e., info that applies to all streams)
# - 'streams', holding a list of dicts, one per stream

# Frame rate metrics:
# - r_frame_rate:
#   - the lowest common multiple of all the frame rates in the stream?
#   - use this one?
# - avg_frame_rate: total # frames / total duration
# https://video.stackexchange.com/questions/20789/ffmpeg-default-output-frame-rate?newreg=e797b27b58a241dc9af8734dc8e14dc4

# The container has 3 streams:
# - codec_type: audio
#       --> no frame rate, nb_frames = number of frames in **audio** stream (right?)
# - codec_type: video
#       --> frame rate as a fraction, nb_frames = total number of frames
#           (from metadata, not by ffmpeg directly decoding every frame)
# - codec_type: 'data'
#       --> 'codec_tag_string': 'tmcd', nb_frames = 1 OJO!
#       --> the timecode stream also contains r_frame_rate and avg_frame_rate
#       --> maybe double-check it matches video?

# %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
# Print timecodes for every frame in a sample video
# ATT!
# - Setting the framerate will automatically set the :attr:`.drop_frame`
#   attribute to correct value.
# - "Frame rates 29.97 and 59.94 are always drop frame" 
#   ---> is this a standard thing? Kinda, in the sense that they are non-integer frame rates
# - I can change the default behaviour w/ force_non_drop_frame (Could be useful? Not for now tho)
video_path = str(list_paths[0])
r_frame_rate_str = timecodes_dict[video_path]['r_frame_rate_str']
n_frames = timecodes_dict[video_path]['n_frames']
start_timecode = timecodes_dict[video_path]['start_timecode']

tc_video_1 = Timecode(r_frame_rate_str, start_timecode)

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
