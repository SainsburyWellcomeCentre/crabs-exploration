import cv2
import matplotlib

matplotlib.use("Agg")  # headless: never connect to X11
from pathlib import Path

import matplotlib.pyplot as plt
import torch
import torchvision.transforms.v2 as transforms

from crabs.detector.models import FasterRCNN

# both assumed to be alongside this file
script_dir = Path(__file__).parent
model_filename = "run_slurm_977884_9.ckpt"
video_filename = "06.09.2023-03-Right-Loop01.mp4"

# --- load model
model = FasterRCNN.load_from_checkpoint(
    str(script_dir / model_filename)
)
model.eval()

# --- grab one frame (or cv2.imread a saved image)
input_video = script_dir / video_filename
cap = cv2.VideoCapture(str(input_video))  # path/to/video.mp4")
cap.set(cv2.CAP_PROP_POS_FRAMES, 100)  # frame index
_, frame = cap.read()
cap.release()
frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

# --- detect
transform = transforms.Compose(
    [
        transforms.ToImage(),
        transforms.ToDtype(torch.float32, scale=True),
    ]
)
img = transform(frame).to(model.device)
with torch.no_grad():
    detections = model([img])

# --- draw boxes as vector graphics over the raw frame & save
score_threshold = 0.0
height, width = frame.shape[:2]
dpi = 300

fig, ax = plt.subplots(figsize=(width / dpi, height / dpi), dpi=dpi)
ax.set_position([0, 0, 1, 1])  # frame fills the figure, no margins
ax.imshow(frame, interpolation="none")  # embed pixels as-is, no resampling
ax.set_axis_off()
ax.set_xlim(0, width)
ax.set_ylim(height, 0)

boxes = detections[0]["boxes"].cpu().numpy()
scores = detections[0]["scores"].cpu().numpy()
for (x1, y1, x2, y2), score in zip(boxes, scores, strict=True):
    if score < score_threshold:
        continue
    ax.add_patch(
        plt.Rectangle(
            (x1, y1),
            x2 - x1,
            y2 - y1,
            fill=False,
            edgecolor="red",
            linewidth=1.5,
        )
    )
    ax.text(x1, y1 - 5, f"{score:.2f}", color="red", fontsize=8)

# vector boxes, lossless frame
fig.savefig(script_dir / f"{input_video.stem}-detections_frame100.svg")
# same, often better for papers
# fig.savefig(script_dir / f"{input_video.stem}-detections_frame100.pdf")
# full-res raster fallback
fig.savefig(
    script_dir / f"{input_video.stem}-detections_frame100.png", dpi=dpi
)
# plt.show()
