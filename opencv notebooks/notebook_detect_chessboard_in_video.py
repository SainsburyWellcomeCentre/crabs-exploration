""" Extract frames in calibration video

"""

# %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
import cv2
from pathlib import Path
import ffmpeg
from timecode import Timecode

# %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
video_path_str = (
    "/Users/sofia/Documents_local/project_Zoo_crabs/crabs-exploration/"
    "stereo calibration/check_chessboard_prelim_230809/"
    "NINJAV__cam1_S001_S001_T001_1.mp4"
)
video_path = Path(video_path_str)


# (i.e., points of intersection of 4 squares, boundaries dont count!)
chessboard_config = {
    "rows": 5,  # 5,  # ATT! THESE ARE INNER POINTS ONLY
    "cols": 8,  # 8,  # ATT! THESE ARE INNER POINTS ONLY
}

video_output_path = video_path.with_name(
    video_path.stem
    + f"_corners_r{chessboard_config['rows']}c{chessboard_config['cols']}_1fps"
    + video_path.suffix
)
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
if s["codec_tag_string"] == "tmcd":
    start_timecode = Timecode(r_frame_rate_str, tmcd_stream["tags"]["timecode"])
    QT_start_timecode = start_timecode * 2 - 1


# %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
# create video capture object
cap = cv2.VideoCapture(video_path_str)
print(cap.isOpened())


# get video params
nframes = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))  # 9000
width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))  # 1920.0
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))  # 1080.0
frame_rate = cap.get(cv2.CAP_PROP_FPS)  # 25fps

print(nframes, width, height, frame_rate)


# %%%%%%%%%%%%%%%%%%%%
# prepare video writer -------
output_frame_rate = 1

# initialise capture and videowriter
cap = cv2.VideoCapture(video_path_str)
fourcc = cv2.VideoWriter_fourcc("m", "p", "4", "v")
videowriter = cv2.VideoWriter(
    str(video_output_path),
    fourcc,
    output_frame_rate,  # frame_rate,
    tuple(x for x in (width, height)),
)

print(cap.get(cv2.CAP_PROP_POS_FRAMES))  # ensure we start at 0

# frames to check
frame_idcs_to_extract = range(0, nframes, 200)
print(f"Total n frames to check: {len(frame_idcs_to_extract)}")

# %%%%%%%%%%%%%%%%%%%%%%%%%%%%
# Try to detect checkerboard in every frame of the video

# count frames added to video
count = 0
count_detected = 0
count_not_detected = 0
# while count <= nframes:
for f_idx in frame_idcs_to_extract:
    # set frame idx
    ret = cap.set(cv2.CAP_PROP_POS_FRAMES, f_idx)
    if not ret:
        print(f"Frame index {f_idx} could not be set")
        continue

    # read frame
    success_frame, frame = cap.read()

    if not success_frame or frame is None:
        print("Can't read frame. Exiting ...")
        break
    else:
        # make grayscale
        img_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        # try to detect chessboard corners
        ret, corners = cv2.findChessboardCorners(
            img_gray,
            (chessboard_config["rows"], chessboard_config["cols"]),
            cv2.CALIB_CB_ADAPTIVE_THRESH
            + cv2.CALIB_CB_FAST_CHECK
            + cv2.CALIB_CB_NORMALIZE_IMAGE,
        )
        # add image with corners if detected, or the og image otherwise
        if ret:
            # add detected corners to the image
            # this modifies the frame 'img' in place OJO!
            corners_img = cv2.drawChessboardCorners(
                frame,
                (chessboard_config["rows"], chessboard_config["cols"]),
                corners,
                ret,
            )
            print(f"Checkerboard detected for frame with idx {f_idx}")
            count_detected += 1

        else:
            corners_img = frame
            print(f"Checkerboard not detected for frame with idx {f_idx}")
            count_not_detected += 1

        # write to video
        videowriter.write(corners_img)

        # update counter
        count += 1

# Release everything if job is finished
cap.release()
videowriter.release()
cv2.destroyAllWindows()

print(video_output_path.name)
print(f"Total frames detected over total processed: {count_detected}/{count}")
print(
    f"Total frames not detected over total processed: {count_not_detected}" f"/{count}"
)


# %%
