# %%
# - timecode
# - ffmpeg python bindings
import pathlib as pl
import cv2
import ffmpeg
from timecode import Timecode

# %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
# Input data
videos_parent_dir = pl.Path(
    "/Users/sofia/Documents_local/project_Zoo_crabs/crabs-exploration/crab_courtyard/"
)

# On file types and timecode
#  - MPEG1/2 timecode is extracted from the GOP, and is available in the video 
#    stream details (-show_streams, see timecode).
#  - MOV timecode is extracted from tmcd track, so is available in the tmcd 
#    stream metadata (-show_streams, see TAG:timecode).
#  - DV, GXF and AVI timecodes are available in format metadata 
#    (-show_format, see TAG:timecode).
file_types = ("**/*.MOV")  #, "**/*.mp4", "**/*.avi")
list_paths = []
for typ in file_types:
    list_paths.extend(
        [p for p in list(videos_parent_dir.glob(typ)) if not p.name.startswith("._")]
    )



# %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
# Get timecode data for each file (using ffmpeg)

def compute_timecode_params_per_video(
        list_paths: list[pl.Path]
):
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
        if tmcd_stream["avg_frame_rate"] != r_frame_rate_str:
            print(f"ERROR: timecode and video frame rates don't match")
            break

        # instantiate a timecode object for this video
        tc_video = Timecode(r_frame_rate_str, start_timecode)

        # save data
        timecodes_dict[video_path] = {
            "r_frame_rate_str": r_frame_rate_str,
            "n_frames": n_frames,
            "start_timecode": start_timecode,
            "timecode_object": tc_video,
        }

    return timecodes_dict


# execute
timecodes_dict = compute_timecode_params_per_video(list_paths)


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

# %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
# Find syncing point

sync_timecode = Timecode(
    '60000/1001', 
    max([vid["start_timecode"] for _, vid in timecodes_dict.items()])
)
min_timecode = Timecode(
    '60000/1001', 
    min([vid["start_timecode"] for _, vid in timecodes_dict.items()])
)

diff_timecode = sync_timecode - min_timecode
diff_in_frames = diff_timecode.tc_to_frames(diff_timecode)  # 190 frames, but only if there was no drop right?
print(diff_in_frames)

for _, vid in timecodes_dict.items():
    if vid["start_timecode"] == sync_timecode:
        vid["opencv_start_idx"] = 0  # opencv uses 0-based index for frames in video capture
    if vid["start_timecode"] == min_timecode:
        vid["opencv_start_idx"] = diff_in_frames


# for vid in timecodes_dict:
#     tc_video = timecodes_dict[str(vid)]["timecode_object"]

#     diff_timecode = Timecode('60000/1001',sync_timecode) - tc_video
#     diff_in_frames = diff_timecode.tc_to_frames(diff_timecode)

#     # sync_frame_from_start_vid = (tc_video.tc_to_frames(sync_timecode) - tc_video.frames) + 1
#     # print(sync_frame_from_start_vid)

# %%%%%%%%%%%%%%%%%%%%%
# Check difference between frames accounts for frame drop
timecode_1 = Timecode('60000/1001', '02:26:59;58')
timecode_2 = Timecode('60000/1001', '02:27:00;04')

diff_timecode = timecode_1 - timecode_2
diff_in_frames = diff_timecode.tc_to_frames(diff_timecode)  # why this odd syntax? can I avoid it
print(diff_in_frames) # it seems as it would be 6 frames between them, but because of frame drop it is actually 2!
# 4 frame number are skipped at the start of every minute
# %%%%%%%%%%%%%%%%%%%%%%%%%%%
# Extract pairs of frames

for video_path, vid in timecodes_dict.items():
    # video_path = list(timecodes_dict.keys())[0]
    print(str(video_path))

    # initialise capture 
    cap = cv2.VideoCapture(str(video_path))
    nframes = cap.get(cv2.CAP_PROP_FRAME_COUNT)  # 9000 --- check it matches?
    print(cap.get(cv2.CAP_PROP_POS_FRAMES))  
    # check initial video index: this points to the next frame to read and is 0-based

    # set index to desired starting reading frame
    print(vid["opencv_start_idx"])
    cap.set(cv2.CAP_PROP_POS_FRAMES, vid["opencv_start_idx"])

    # create output dir
    output_dir_one_camera = pl.Path('calibration_pairs') / pl.Path(video_path).stem 
    output_dir_one_camera.mkdir(parents=True, exist_ok=True)

    pair_count = 1
    for frame_idx0 in range(
        vid["opencv_start_idx"], 
        vid["opencv_start_idx"]+3
    ):  # vid["n_frames"]+1): ---------
        # print index (0-based)
        print(cap.get(cv2.CAP_PROP_POS_FRAMES))  

        # read frame
        success, frame = cap.read()

        # write frame to file
        if success:
            file_path = (
                output_dir_one_camera / 
                f"frame_{frame_idx0+1}_pair_{pair_count}.png"
            )
            # here we are getting the 'next' index right? so name would be 1-based?
            flag_saved = cv2.imwrite(
                    str(file_path),
                    frame
            )  # save frame as JPEG file 

            if flag_saved:
                print(f"frame {frame_idx0} saved at {file_path}")
            else:
                print(f"ERROR saving {pl.Path(video_path).stem}, frame {frame_idx0}...skipping")
                continue

            # increase pair count
            pair_count +=1



# %%
