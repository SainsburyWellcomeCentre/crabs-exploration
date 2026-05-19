"""Run SAM3 image inference on burrow frames using bbox/point prompts.

Follows the official SAM3 image predictor example:
https://github.com/facebookresearch/sam3/blob/main/examples/sam3_image_predictor_example.ipynb

Prompt data is produced upstream (one CSV per video, grouped by ``group_id``)
with columns:
  prompt_point_x, prompt_point_y,
  prompt_bbox_xmin, prompt_bbox_ymin, prompt_bbox_xmax, prompt_bbox_ymax
"""

# %%
# Imports
import os
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from PIL import Image

import sam3
from sam3 import build_sam3_image_model
from sam3.model.box_ops import box_xywh_to_cxcywh
from sam3.model.sam3_image_processor import Sam3Processor
from sam3.visualization_utils import draw_box_on_image, normalize_bbox, plot_results

sam3_root = os.path.join(os.path.dirname(sam3.__file__), "..")

# %%
# Input data
# - images_dir: directory of PNG frames
# - prompt_coords_dir: directory of per-video prompt CSVs

images_dir = (
    "/Users/sofia/arc/project_Zoo_crabs/crab_loops_end_frames_slurm2764495"
)
prompt_coords_dir = "/Users/sofia/arc/project_Zoo_crabs/burrow_prompts_slurm_3012602/coords_20260519_105922"

# Prediction params
TEXT_PROMPT = "burrow"  # set to None to skip the text prompt
CONF_THRESHOLD = 0.5

# Output dir for masks (.npz per frame)
OUTPUT_DIR = Path("./output_burrows_sam3")


# %%%%%%%%%%
# Helpers
class ImageArrayLazy:
    """A lazy array for images in a list."""

    def __init__(self, img_paths):
        self.img_paths = sorted(img_paths)
        # add image shape, assuming all have same as
        # first sample
        sample = np.array(Image.open(img_paths[0]))  # H, W, C
        self.img_h, self.img_w, self.img_c = sample.shape

    def __len__(self):
        return len(self.img_paths)

    def __getitem__(self, idx):
        return np.array(Image.open(self.img_paths[idx]))

    @property
    def shape(self):
        return (len(self.img_paths), self.img_h, self.img_w, self.img_c)
        # B, H, W, C


# %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
# Load frames as a lazy array and map each frame to its video / date

list_image_files = sorted(list(Path(images_dir).glob("*.png")))
image_array = ImageArrayLazy(list_image_files)
print(image_array.shape)

list_video_per_img = [
    img.stem.split("_", 1)[0].split("-Loop")[0]
    for img in image_array.img_paths
]
list_date_per_img = [video.split("-")[0] for video in list_video_per_img]


# %%%%%%%%%%%%%%%%%%%%%%%%%%
# Load prompts (one CSV per video, concatenated)

list_prompt_csv = sorted(list(Path(prompt_coords_dir).glob("*.csv")))
df_prompts = pd.concat([pd.read_csv(f) for f in list_prompt_csv])

# bbox prompts in pixel xyxy, keyed by group_id (== video string)
bboxes_xyxy_per_video = {
    key: group[
        [
            "prompt_bbox_xmin",
            "prompt_bbox_ymin",
            "prompt_bbox_xmax",
            "prompt_bbox_ymax",
        ]
    ].to_numpy()
    for key, group in df_prompts.groupby("group_id")
}


# %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
# Build SAM3 image model and processor

# turn on tfloat32 for Ampere GPUs
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True
if torch.cuda.is_available():
    torch.autocast("cuda", dtype=torch.bfloat16).__enter__()

bpe_path = f"{sam3_root}/assets/bpe_simple_vocab_16e6.txt.gz"
model = build_sam3_image_model(bpe_path=bpe_path)
processor = Sam3Processor(model, confidence_threshold=CONF_THRESHOLD)


# %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
# Run inference on every frame using its video's bbox prompts

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
masks_per_frame = {}

for frame_idx in range(len(image_array)):
    video_str = list_video_per_img[frame_idx]
    boxes_xyxy = bboxes_xyxy_per_video.get(video_str)
    if boxes_xyxy is None or len(boxes_xyxy) == 0:
        print(f"Frame {frame_idx} ({video_str}): no prompts, skipping")
        continue

    image = Image.fromarray(image_array[frame_idx])
    width, height = image.size
    inference_state = processor.set_image(image)
    processor.reset_all_prompts(inference_state)

    # optional text prompt
    if TEXT_PROMPT is not None:
        inference_state = processor.set_text_prompt(
            state=inference_state, prompt=TEXT_PROMPT
        )

    # bbox exemplars: xyxy (px) -> xywh -> cxcywh -> normalized
    boxes_xywh = boxes_xyxy.astype(np.float32).copy()
    boxes_xywh[:, 2] -= boxes_xywh[:, 0]  # w = xmax - xmin
    boxes_xywh[:, 3] -= boxes_xywh[:, 1]  # h = ymax - ymin
    boxes_cxcywh = box_xywh_to_cxcywh(
        torch.tensor(boxes_xywh).view(-1, 4)
    )
    norm_boxes_cxcywh = normalize_bbox(boxes_cxcywh, width, height).tolist()

    for box in norm_boxes_cxcywh:
        inference_state = processor.add_geometric_prompt(
            state=inference_state, box=box, label=True
        )

    results = processor.get_results(inference_state)
    masks_per_frame[frame_idx] = results

    # TODO: save as a zarr store?
    np.savez_compressed(
        OUTPUT_DIR / f"frame_{frame_idx:06d}.npz",
        masks=results["masks"],
        scores=results["scores"],
        video=video_str,
    )
    print(
        f"Frame {frame_idx} ({video_str}): "
        f"{len(results['masks'])} masks"
    )


# %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
# Visualise one frame: prompt boxes + predicted masks

frame_idx = next(iter(masks_per_frame))
video_str = list_video_per_img[frame_idx]
image = Image.fromarray(image_array[frame_idx])

image_with_boxes = image
for x1, y1, x2, y2 in bboxes_xyxy_per_video[video_str]:
    image_with_boxes = draw_box_on_image(
        image_with_boxes, [x1, y1, x2 - x1, y2 - y1], (0, 255, 0)
    )

plt.figure()
plt.imshow(image_with_boxes)
plt.axis("off")
plt.title(f"frame {frame_idx} ({video_str}) - prompt boxes")
plt.show()

# rerun the predictor on this frame to plot SAM3 results overlaid
inference_state = processor.set_image(image)
processor.reset_all_prompts(inference_state)
if TEXT_PROMPT is not None:
    inference_state = processor.set_text_prompt(
        state=inference_state, prompt=TEXT_PROMPT
    )
boxes_xyxy = bboxes_xyxy_per_video[video_str].astype(np.float32).copy()
boxes_xywh = boxes_xyxy.copy()
boxes_xywh[:, 2] -= boxes_xywh[:, 0]
boxes_xywh[:, 3] -= boxes_xywh[:, 1]
boxes_cxcywh = box_xywh_to_cxcywh(torch.tensor(boxes_xywh).view(-1, 4))
for box in normalize_bbox(
    boxes_cxcywh, *image.size
).tolist():
    inference_state = processor.add_geometric_prompt(
        state=inference_state, box=box, label=True
    )
plot_results(image, inference_state)

# %%
