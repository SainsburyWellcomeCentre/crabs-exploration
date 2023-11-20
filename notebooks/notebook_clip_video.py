# %%
from pathlib import Path

import cv2

# %%%%%%%%%%%%%%%%%%%%%%%%%%%%%
# Save a clip of selected file

input_video_file = Path(
    "/Users/sofia/Documents_local/project_Zoo_crabs/crabs-exploration/tests/data/"
    "NINJAV_S001_S001_T003_subclip.mp4",
)

clip_suffix = "p1_05s"

output_dir = "/Users/sofia/Documents_local/project_Zoo_crabs/crabs-exploration/tests/data"

# %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
# Get video params
cap = cv2.VideoCapture(str(input_video_file))
print(cap.isOpened())

# get video params
nframes = cap.get(cv2.CAP_PROP_FRAME_COUNT)  # 9000
width = cap.get(cv2.CAP_PROP_FRAME_WIDTH)  # 1920.0
height = cap.get(cv2.CAP_PROP_FRAME_HEIGHT)  # 1080.0
frame_rate = cap.get(cv2.CAP_PROP_FPS)  # 25fps

size = (int(width), int(height))

start_frame = 0  # 10*frame_rate
end_frame = start_frame + 0.5 * frame_rate  # 22500

# %%%%%%%%%%%%%%%%%%%%%%%%%%%%%
# initialise capture and videowriter
# cap = cv2.VideoCapture(str(video_file))
fourcc = cv2.VideoWriter_fourcc("m", "p", "4", "v")
videowriter = cv2.VideoWriter(
    str(output_dir / Path(input_video_file.stem + f"_{clip_suffix}.mp4")),
    fourcc,
    frame_rate,
    size,
)

print(cap.get(cv2.CAP_PROP_POS_FRAMES))  # ensure we start at 0

# %%%%%%%%%%%%%%%%%%%

print(cap.get(cv2.CAP_PROP_POS_FRAMES))  # ensure we start at 0
print(end_frame)

count = 0
while count <= end_frame:
    # read frame
    success_frame, frame = cap.read()

    # if not successfully read, exit
    if not success_frame:
        print("Can't receive frame. Exiting ...")
        break

    # if frame within clip bounds: write to video
    if (count >= start_frame) and (count <= end_frame):
        videowriter.write(frame)

    count += 1

# Release everything if job is finished
cap.release()
videowriter.release()
cv2.destroyAllWindows()
# %%
