""" Extract frames in calibration video

"""

# %%
from pathlib import Path

import cv2

# import timecode
import ffmpeg
from timecode import Timecode

# %%
video_path_str = (
    "/Users/sofia/Documents_local/project_Zoo_crabs/crabs-exploration/"
    "stereo calibration/check_chessboard_prelim_230809/"
    "NINJAV__cam1_S001_S001_T001_1.mp4"
)
video_path = Path(video_path_str)
video_output_dir = Path(
    "/Users/sofia/Documents_local/project_Zoo_crabs/crabs-exploration"
    "/stereo calibration/check_chessboard_prelim_230809"
)


# QuickTime timecodes
# QT_start_timecode = start_timecode*2 - 1
QT_timecodes_to_extract = [
    "04:53:27:24",
    "04:54:07:19",
    "04:54:21:15",
    "04:54:30:08",
    "04:55:07:29",
    "04:55:12:13",  # ---- none of these are detected
    "04:53:52:07",
    "04:53:43:01",
    "04:54:37:30",
    "04:54:40:44",
    "04:55:20:28",
    "04:55:32:32",
    "04:55:43:26",
    "04:55:50:24",
]

# (i.e., points of intersection of 4 squares, boundaries dont count!)
chessboard_config = {
    "rows": 6,  # ATT! THESE ARE INNER POINTS ONLY
    "cols": 9,  # ATT! THESE ARE INNER POINTS ONLY
}

# %%%%%%%%%%%%%%%%%%%%%%%%%%%%
# Extract frame idcs from timecodes
# --- not useful bc QT doesn't actually show timecodes

ffprobe_json = ffmpeg.probe(video_path_str)

# parse streams
# we assume one video stream only and one timecode stream
for s in ffprobe_json["streams"]:
    if s["codec_type"] == "video":
        video_stream = s
    if s["codec_tag_string"] == "tmcd":
        tmcd_stream = s

# get frame rate
r_frame_rate_str = video_stream["r_frame_rate"]

# get start timecode
start_timecode = Timecode(r_frame_rate_str, tmcd_stream["tags"]["timecode"])
QT_start_timecode = start_timecode * 2 - 1

frame_idcs_to_extract = []
for tc in QT_timecodes_to_extract:
    QT_curr_timecode = Timecode(r_frame_rate_str, tc)
    if QT_curr_timecode == QT_start_timecode:
        frame_idcs_to_extract.append(0)
    else:
        diff_timecode = QT_curr_timecode - QT_start_timecode
        frame_idcs_to_extract.append(diff_timecode.frames)

# frame_idcs_to_extract = []
# for tc in QT_timecodes_to_extract:
#     QT_curr_timecode = Timecode(r_frame_rate_str, tc)
#     curr_timecode = (QT_curr_timecode+1)/2
#     if curr_timecode == start_timecode:
#         frame_idcs_to_extract.append(0)
#     else:
#         diff_timecode = curr_timecode - start_timecode
#         frame_idcs_to_extract.append(diff_timecode.frames)

# %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
# create video capture object
cap = cv2.VideoCapture(video_path_str)
print(cap.isOpened())


# get video params
nframes = cap.get(cv2.CAP_PROP_FRAME_COUNT)  # 9000
width = cap.get(cv2.CAP_PROP_FRAME_WIDTH)  # 1920.0
height = cap.get(cv2.CAP_PROP_FRAME_HEIGHT)  # 1080.0
frame_rate = cap.get(cv2.CAP_PROP_FPS)  # 25fps

print(nframes, width, height, frame_rate)


# %%%%%%%%%%%%%%%%%%%%%%%%%%%%
# Extract selected frame
for f_idx in frame_idcs_to_extract:
    ret = cap.set(cv2.CAP_PROP_POS_FRAMES, f_idx)
    if not ret:
        print(f"Frame index {f_idx} could not be set")
        continue
    success, frame = cap.read()

    # save to file
    if not success or frame is None:
        raise KeyError(
            f"Unable to load frame {f_idx} from {(video_path).stem}."
        )

    else:
        file_path = video_output_dir / Path(
            f"{video_path.parent.stem}_"
            f"{video_path.stem}_"
            f"frame_{f_idx:06d}.png"
        )
        img_saved = cv2.imwrite(str(file_path), frame)
        if img_saved:
            print(f"frame {f_idx} saved at {file_path}")
        else:
            print(
                f"ERROR saving {(video_path).stem}, "
                "frame {f_idx}...skipping"
            )
            continue


# close capture
cap.release()
# %%%%%%%%%%%%%%%%%%%%%%%%%%%
# try to detect checkerboard corners in extracted frames
# -------
video_output_dir = Path(
    "/Users/sofia/Documents_local/project_Zoo_crabs/crabs-exploration/"
    "stereo calibration/"
    "check_chessboard_prelim_ramalhete23/2023-8-8_resolution-test"
)
detected_frames_subdir = video_output_dir.parent / "detected"
# -------
list_frames = [
    f for f in video_output_dir.glob("*.png") if not f.name.startswith("._")
]

# (i.e., points of intersection of 4 squares, boundaries dont count!)
chessboard_config = {
    "rows": 5,  # ATT! THESE ARE INNER POINTS ONLY
    "cols": 8,  # ATT! THESE ARE INNER POINTS ONLY
}


for file in list_frames:
    # read file
    img = cv2.imread(
        str(file)
    )  # size_og = (img_og.shape[1], img_og.shape[0]) # we want width, height!

    # make grayscale
    img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # try to detect chessboard corners
    ret, corners = cv2.findChessboardCorners(
        img_gray,
        (chessboard_config["rows"], chessboard_config["cols"]),
        cv2.CALIB_CB_ADAPTIVE_THRESH
        + cv2.CALIB_CB_FAST_CHECK
        + cv2.CALIB_CB_NORMALIZE_IMAGE,
    )

    # save with corners if detected, or the og image otherwise
    if ret:
        # add detected corners to the image
        # this modifies the frame 'img' in place OJO!
        corners_img = cv2.drawChessboardCorners(
            img,
            (chessboard_config["rows"], chessboard_config["cols"]),
            corners,
            ret,
        )
        print(f"Checkerboard detected for {file.stem}.png")

        file_path = (
            detected_frames_subdir  # video_output_dir.parent / "detected"
            / f"{file.stem}_checkerboard.png"
        )
    else:
        corners_img = img
        print(f"Checkerboard not detected for {file.stem}.png")

        file_path = (
            detected_frames_subdir  # video_output_dir.parent / "detected"
            / f"{file.stem}_no_checkerboard.png"
        )

    # save
    flag_saved = cv2.imwrite(str(file_path), corners_img)
    if flag_saved:
        print(f"{file_path.stem} saved")
    else:
        print(f"ERROR saving {file_path.stem} " "...skipping")
        continue

# %%
