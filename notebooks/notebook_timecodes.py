# %%
from pathlib import Path

import cv2
import matplotlib.pyplot as plt

import crabs.stereo_calibration.extract_pairs_of_frames as stereo

# %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
# Input data
input_videos_parent_dir = (
    "/Users/sofia/Documents_local/project_Zoo_crabs/data/crab_courtyard/"
)
video_extensions = ["MOV"]
output_calibration_dir = "./calibration_pairs"

# Transform file_types
file_types = tuple(f"**/*.{ext}" for ext in video_extensions)

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

# Chessboard parameters
# assuming rows < cols?
chessboard_config = {
    "rows": 6,  # ATT! THESE ARE INNER POINTS ONLY (i.e., points of intersection of 4 squares, boundaries dont count!)
    "cols": 9,  # ATT! THESE ARE INNER POINTS ONLY
}

# %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
# Extract timecode params
timecodes_dict = stereo.compute_timecode_params_per_video(list_paths)

# NOTE!
# (timecode_v1['start_timecode'] - timecode_v1[ky0]['end_timecode']).frames = nframes-1
# - timecodes are frame labels
# - subtraction of timecodes is commutative

# %%%%%%%%%%%%%%%%%%%%%%%%%%%
# Compute syncing timecodes: the max start timecode across all videos
# and the min end timecode
max_start_timecode, min_end_timecode = stereo.compute_synching_timecodes(
    timecodes_dict
)

# %%%%%%%%%%%%%%%%%%%%%%%%%%%%%
# Compute opencv start index per video:
timecodes_dict = stereo.compute_opencv_start_idx(
    timecodes_dict, (max_start_timecode, min_end_timecode)
)

# %%%%%%%%%%%%%%%%%%%%%%%%%%%%%
# Extract frames with opencv and save to directory
for vid_str, vid_dict in timecodes_dict.items():
    stereo.extract_chessboard_frames_from_video(
        vid_str,
        vid_dict,
        chessboard_config,
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
    "start_timecode"
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

#  %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
# Check if chessboard is detected
video_path_str = list(timecodes_dict.keys())[0]
# opencv_start_idx = timecodes_dict[video_path_str]['opencv_start_idx']
opencv_end_idx = timecodes_dict[video_path_str]["opencv_end_idx"]

cap = cv2.VideoCapture(video_path_str)
# there should be one around 13s in for Camera2 video
frame_w_chessboard_0idx = int(11 * 59.94) - 1  # int(13*59.94) - 1
frame_wo_chessboard_0idx = int(3 * 59.94) - 1
cap.set(cv2.CAP_PROP_POS_FRAMES, frame_w_chessboard_0idx)

# read frames
for kk in range(timecodes_dict[video_path_str]["n_frames"]):
    frame_idx0 = cap.get(cv2.CAP_PROP_POS_FRAMES)
    print(frame_idx0)

    chessboard_tuple = (chessboard_config["cols"], chessboard_config["rows"])
    # ---> the order of these is unclear, does it matter?

    success, frame = cap.read()
    if success:
        # ---------------
        # Find the chessboard corners
        # If desired number of corners are found in the image then ret = true
        # TODO: append 2d coords of corners?
        frame_gray = cv2.cvtColor(
            frame, cv2.COLOR_BGR2GRAY
        )  # not sure if this actually makes it easieror not
        # but all examples that I've seen do it

        # find chessboard
        # to check flags:
        # https://docs.opencv.org/4.x/d9/d0c/group__calib3d.html#ga93efa9b0aa890de240ca32b11253dd4a
        # TODO: the following is very slow when no chessboard is present
        ret, corners = cv2.findChessboardCorners(
            frame_gray,
            chessboard_tuple,
            flags=(
                cv2.CALIB_CB_ADAPTIVE_THRESH
                + cv2.CALIB_CB_FAST_CHECK
                + cv2.CALIB_CB_NORMALIZE_IMAGE
            ),
        )

        # ----- this should be faster but doesnt work at all for me!!!
        # checkChessboard is faster if no chessboard is present?
        # https://shimat.github.io/opencvsharp_2410/html/7b6a746d-3776-3ae7-ab62-4b17f31bab30.htm
        # but it doesnt work :( it detects chessboards in all of them
        #
        # ret = cv2.checkChessboard(
        #     frame_gray,
        #     (chessboard_config['cols'], chessboard_config['rows']),
        # )
        # -------------

        # inspect corners
        # - corners is a numpy array of size (nrows*ncols, 1, 2)
        # - why the middle single dimension?? who knows...
        # - these are pixel coords relative to top left of the image? (size 4096x2160 for Cam2)
        # - the first corner is the top right of the checkerboard and then goes along columns?
        #
        # from the tutorial....
        # - opencv takes  the bottom left(??) of the checkerboard as the origin (I think)
        # - This means each frame we use to calibrate gets a separate origin

        if ret:
            print(
                "Chessboard detected on"
                f" {Path(video_path_str).stem}, "
                f"frame {frame_idx0}"
            )

            plt.figure()
            plt.imshow(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
            # or: frame[:, :, ::-1]) #::-1 changing BGR to RGB
            # doesnt change the order of the channels it in place
            nsamples = 2
            plt.scatter(
                x=corners[:nsamples, 0, 0],
                y=corners[:nsamples, 0, 1],
                s=1,
                c=range(corners[:nsamples, :, :].shape[0]),
            )

            # usually people will refine the corners
            # corners_img = cv2.drawChessboardCorners(
            #     frame,
            #     chessboard_tuple,
            #     corners,
            #     ret
            # ) #--- this modifies frame in place OJO! it adds markers for the detected corners

            # cv2.imshow('img', corners_img) #frame_gray)
            # cv2.waitKey(0)
            # cv2.destroyAllWindows()
            # cv2.waitKey(1) # click any key to close while window is active

            break
        else:
            print(
                "WARNING: No chessboard detected on"
                f" {Path(video_path_str).stem}, "
                f"frame {frame_idx0}...skipping"
            )
        # print(ret)
        # print(corners)
# %%
