# %%
# - timecode
# - ffmpeg python bindings
import ffmpeg
from timecode import Timecode

# %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
# Input data
video_path = '/Users/sofia/Documents_local/project_Zoo_crabs/crabs-exploration/crab_courtyard/Camera1_NINJAV_S001_S001_T011.MOV'

# %%%%%%%%%%%%%%%%%%%%%%%%%%
# Get start timecode for each file (using ffmpeg)
# dict for each file: frame_rate and start code

output_json = ffmpeg.probe(video_path)
# output is a dict w/ two fields: 
# - 'format', holds container-level info (i.e., info that applies to all streams)
# - 'streams', holding a list of dicts, one per stream

# format -- also from data steam
start_timecode = output_json['format']['tags']['timecode']

# streams
video_stream = [
    s 
    for s in output_json['streams']
    if s['codec_type'] == 'video'
][0]  # we assume one video stream

r_frame_rate_str = video_stream['r_frame_rate'] 
n_frames = int(video_stream['nb_frames'])


# Frame rate metrics:
# - r_frame_rate: the lowest common multiple of all the frame rates in the stream?
#  --- use this one?
# - avg_frame_rate: total # frames / total duration
# https://video.stackexchange.com/questions/20789/ffmpeg-default-output-frame-rate?newreg=e797b27b58a241dc9af8734dc8e14dc4

# The container has 3 streams:
# - codec_type: audio --> no frame rate, nb_frames = number of frames in audio stream?
# - codec_type: video ---> w/ frame rate as a fraction, nb_frames = total number of frames (from metadata, not by ffmpeg directly decoding every frame)
# - codec_type: 'data' ---> 'codec_tag_string': 'tmcd', nb_frames = 1 OJO!
#   - the timecode stream also contains r_frame_rate and avg_frame_rate 
#       --> maybe double-check it matches video?

# %%%%%%%%%%%%%%%%%%%%%%%%%
# Print timecodes for video?
# ATT!
# - Setting the framerate will automatically set the :attr:`.drop_frame`
#   attribute to correct value. 
# - "Frame rates 29.97 and 59.94 are always drop frame" ---> is this a standard thing?
# - I can change the default behaviour w/ force_non_drop_frame
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
        *tc_video_1.frames_to_tc(
            curr_frame_number
        )
    )

    print(
        f"Frame {curr_frame_number - tc_video_1.frames + 1}"
        f" \t Timecode {current_frame_timecode}")



# .frame_number property
# methods:
# - frames_to_tc
# - tc_to_frames

# Drop frame?

# %%
