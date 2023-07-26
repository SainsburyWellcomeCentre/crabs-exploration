# %%
import cv2
from pathlib import Path

# %%
video_path = "/Users/sofia/Documents_local/project_Zoo_crabs/opencv notebooks/cam2_NINJAV_S001_S001_T004_out_domain_sample.mp4"

# %%
# initialise video capture
cap = cv2.VideoCapture(str(video_path))

nframes = cap.get(cv2.CAP_PROP_FRAME_COUNT)  # 9000
width = cap.get(cv2.CAP_PROP_FRAME_WIDTH)  # 1920.0
height = cap.get(cv2.CAP_PROP_FRAME_HEIGHT)  # 1080.0
frame_rate = cap.get(cv2.CAP_PROP_FPS)  # 25fps

# %%
# initialise video writer
videowriter = cv2.VideoWriter(
    f"{Path(video_path).stem}_flow.mp4",
    cv2.VideoWriter_fourcc("m", "p", "4", "v"),
    frame_rate,
    tuple(int(x) for x in (width, height)),
)

# %%
frame_idx = 25  # to read frame 25 (w/ idx-25 in zero indexed system)

cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
print(cap.get(cv2.CAP_PROP_POS_FRAMES))

# read selected frame
success_frame_1, frame_1 = cap.read()

# check next index
print(cap.get(cv2.CAP_PROP_POS_FRAMES))
# %%
