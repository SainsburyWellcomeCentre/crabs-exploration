"""Run SAM3 image inference on burrow frames using bbox/point prompts.

Follows the official SAM3 image predictor example:
https://github.com/facebookresearch/sam3/blob/main/examples/sam3_image_predictor_example.ipynb
and https://github.com/facebookresearch/sam3#basic-usage  


Prompt data is produced upstream (one CSV per video, grouped by ``group_id``)
with columns:
  prompt_point_x, prompt_point_y,
  prompt_bbox_xmin, prompt_bbox_ymin, prompt_bbox_xmax, prompt_bbox_ymax
"""
# /// script
# requires-python = ">=3.11"
# dependencies = [
#   "Pillow",
#   "zarr",
#   "torch>=2.5.1",
#   "torchvision>=0.20.1",
#   "sam3 @ git+https://github.com/facebookresearch/sam3.git",
#   "einops",
#   "huggingface_hub",
#   "ipympl",
# ]
#
# [tool.uv.sources]
# torch = { index = "pytorch-cu128" }
# torchvision = { index = "pytorch-cu128" }
#
# [[tool.uv.index]]
# name = "pytorch-cu128"
# url = "https://download.pytorch.org/whl/cu128"
# explicit = true
# ///

# %%
# Imports
import os

# Reduce CUDA allocator fragmentation. Must be set before torch is
# imported -> restart the kernel for this to take effect in a notebook.
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

from datetime import datetime
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import zarr
from PIL import Image
from sam3 import build_sam3_image_model
from sam3.model.box_ops import box_xywh_to_cxcywh
from sam3.model.sam3_image_processor import Sam3Processor
from sam3.visualization_utils import (
    draw_box_on_image,
    normalize_bbox,
)

# sam3_root = os.path.join(os.path.dirname(sam3.__file__), "..")

# %%
# Input data
# - images_dir: directory of PNG frames
# - prompt_coords_dir: directory of per-video prompt CSVs

images_dir = "/home/sminano/swc/project_crabs/burrow_mean_image_slurm_3013437"
prompt_coords_dir = "/home/sminano/swc/project_crabs/burrow_prompts_per_day_20260423_143244"
#"/home/sminano/swc/project_crabs/burrow_prompts_slurm_3012602/coords_20260519_105922"

flag_using_date_prompts = True

# Prediction params
TEXT_PROMPT = "burrow"  # set to None to skip the text prompt
CONF_THRESHOLD = 0.5

# Output dir for masks 
# TODO: add timestamp
OUTPUT_DIR = Path("/home/sminano/swc/project_crabs/crabs-exploration/output_burrows_sam3")

# use only the top 3 prompt boxes per video (or date)
# (they should be sorted by peak height)
# top_n_bboxes = 10



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


def create_mask_zarr(path_to_zarr, zarr_array_shape, metadata_dict=None):
    """Create a zarr store for ID-encoded masks and write metadata."""
    n_images, image_h, image_w = zarr_array_shape[:3]
    mask_zarr = zarr.open(
        path_to_zarr,
        mode="w",
        shape=(n_images, image_h, image_w),
        dtype="int16",
        fill_value=0,
        chunks=(1, image_h, image_w),
    )
    if metadata_dict is not None:
        mask_zarr.attrs.update(metadata_dict)
    return mask_zarr


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


# %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
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

# # turn on tfloat32 for Ampere GPUs
# torch.backends.cuda.matmul.allow_tf32 = True
# torch.backends.cudnn.allow_tf32 = True

# to avoid bfloat16 and float mismatch
if torch.cuda.is_available():
    torch.autocast("cuda", dtype=torch.bfloat16).__enter__()
# torch.autocast("cuda", dtype=torch.bfloat16).__enter__()

# %%

# bpe_path = f"{sam3_root}/assets/bpe_simple_vocab_16e6.txt.gz"
model = build_sam3_image_model()

# # Fix dtype mismatch issue by ensuring all parameters are float32
# model = model.float()
# for name, param in model.named_parameters():
#     param.data = param.data.float()
# for name, buffer in model.named_buffers():
#     if buffer.dtype != torch.complex64:  # Keep complex buffers as-is (for rotary embeddings)
#         buffer.data = buffer.data.float()

processor = Sam3Processor(model, confidence_threshold=CONF_THRESHOLD)


# %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
# Initialise the output ID-encoded mask zarr store

timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
output_masks_zarr = OUTPUT_DIR / f"masks_{timestamp}.zarr"

n_images, image_h, image_w = image_array.shape[:3]
metadata_dict = {
    "timestamp": timestamp,
    "sam3_model": "sam3_image",
    "source_images_dir": str(images_dir),
    "prompt_coords_dir": str(prompt_coords_dir),
    "text_prompt": TEXT_PROMPT,
    "confidence_threshold": CONF_THRESHOLD,
    "n_images": n_images,
    "image_shape": [image_h, image_w],
    "prompt_type": "bounding_box",
    "mask_encoding": "instance_id",
    "background_label": 0,
    "id_offset": 1,
}
mask_zarr = create_mask_zarr(
    output_masks_zarr,
    (n_images, image_h, image_w),
    metadata_dict=metadata_dict,
)
 # %%
%matplotlib widget
# %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
# Interactively select which prompt boxes to pass to SAM3.
# Click inside a box to toggle it: lime = selected, red = excluded.

import matplotlib.patches as patches

select_frame_idx = 0
if flag_using_date_prompts:
    select_video_str = list_date_per_img[select_frame_idx]
else:
    select_video_str = list_video_per_img[select_frame_idx]

boxes_xyxy = bboxes_xyxy_per_video[select_video_str]
selected = np.zeros(len(boxes_xyxy), dtype=bool)  # start with all unselected

# lime - unselected
# red - selected

fig, ax = plt.subplots()
ax.imshow(image_array[select_frame_idx])
ax.set_axis_off()

rects = []
for x1, y1, x2, y2 in boxes_xyxy:
    r = patches.Rectangle(
        (x1, y1), x2 - x1, y2 - y1,
        fill=False, linewidth=2, edgecolor="lime",
    )
    ax.add_patch(r)
    rects.append(r)

def _refresh_title():
    ax.set_title(
        f"{select_video_str}: {selected.sum()}/{len(selected)} boxes selected"
    )

def _on_click(event):
    if event.inaxes != ax:
        return
    for i, (x1, y1, x2, y2) in enumerate(boxes_xyxy):
        if x1 <= event.xdata <= x2 and y1 <= event.ydata <= y2:
            selected[i] = not selected[i]
            rects[i].set_edgecolor("lime" if not selected[i] else "red")
            break  # first match only — minimal handling of overlaps
    _refresh_title()
    fig.canvas.draw_idle()

_refresh_title()
fig.canvas.mpl_connect("button_press_event", _on_click)
plt.show()


# %%
# once you're happy with the selection), commit the choice back so the inference loop picks it up unchanged:
# Apply the selection — the inference loop reads bboxes_xyxy_per_video
bboxes_xyxy_per_video[select_video_str] = boxes_xyxy[selected]
print(f"{select_video_str}: kept {selected.sum()} boxes")


# %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
# Run inference on every frame and write ID-encoded masks to zarr

processed_frames = []

for frame_idx in range(len(image_array)):
    # with torch.autocast("cuda", dtype=torch.bfloat16):

    # Get corresponding video
    if flag_using_date_prompts:
        video_str = list_date_per_img[frame_idx]
    else:
        video_str = list_video_per_img[frame_idx]

    # Get prompts for that video
    boxes_xyxy = bboxes_xyxy_per_video.get(video_str)
    if boxes_xyxy is None or len(boxes_xyxy) == 0:
        print(f"Frame {frame_idx} ({video_str}): no prompts, skipping")
        continue

    # Load image
    image = Image.fromarray(image_array[frame_idx])
    width, height = image.size
    inference_state = processor.set_image(image) 
    # maybe: set_image_batch?
    processor.reset_all_prompts(inference_state) # mutates the state dict in place

    # optional text prompt
    if TEXT_PROMPT is not None:
        inference_state = processor.set_text_prompt(
            state=inference_state, prompt=TEXT_PROMPT
        )

    # Express bboxes for exemplars as norm_cxcywh
    # bbox exemplars: xyxy (pixels) -> xywh -> cxcywh -> normalized
    boxes_xywh = boxes_xyxy.astype(np.float32).copy()
    boxes_xywh[:, 2] -= boxes_xywh[:, 0]  # w = xmax - xmin
    boxes_xywh[:, 3] -= boxes_xywh[:, 1]  # h = ymax - ymin
    boxes_cxcywh = box_xywh_to_cxcywh(torch.tensor(boxes_xywh).view(-1, 4))
    norm_boxes_cxcywh = normalize_bbox(boxes_cxcywh, width, height).tolist()

    # Add the top N bboxes as a prompt
    for box in norm_boxes_cxcywh:
        inference_state = processor.add_geometric_prompt(
            state=inference_state, box=box, label=True
        )

    # add_geometric_prompt / set_text_prompt run inference internally
    # and return the updated state; predictions live in the state dict
    # under the "masks" / "boxes" / "scores" keys (no get_results method)

    # Express results as an ID-encoded mask
    # boolean masks (N, H, W) -> ID-encoded (H, W); higher ID wins on overlap
    # Move masks to CPU, then release this frame's GPU state before the
    # next frame: otherwise the previous state (backbone features +
    # full-res masks_logits) stays alive during the next forward pass.
    masks = inference_state["masks"].cpu().numpy()
    del inference_state
    torch.cuda.empty_cache()
    
    masks = masks.squeeze(1) if masks.ndim == 4 else masks  # (N, H, W)
    n_objects = masks.shape[0]
    if n_objects == 0:
        print(f"Frame {frame_idx} ({video_str}): no detections")
        continue

    obj_ids = np.arange(1, n_objects + 1, dtype=np.int16)[:, None, None]
    id_mask = (masks.astype(bool) * obj_ids).max(axis=0)

    mask_zarr[frame_idx] = id_mask
    processed_frames.append(frame_idx)
    mask_zarr.attrs["annotated_frames"] = processed_frames
    print(f"Frame {frame_idx} ({video_str}): {n_objects} masks")

print(f"Saved ID-encoded mask zarr to {output_masks_zarr}")


# %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
# Visualise one frame: prompt boxes + predicted masks

frame_idx = 0 #processed_frames[0]
if flag_using_date_prompts:
    video_str = list_date_per_img[frame_idx]
else:
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

# Plot the ID-encoded masks read back from the zarr store
# TODO: why only 3 masks?
id_mask = mask_zarr[frame_idx]  # (H, W), 0 = background
masked = np.ma.masked_where(id_mask == 0, id_mask)

plt.figure()
plt.imshow(image)
plt.imshow(masked, cmap="tab10", alpha=0.5, interpolation="nearest")
plt.axis("off")
plt.title(f"frame {frame_idx} ({video_str}) - SAM3 masks (from zarr)")
plt.show()

# TODO add count of predicted masks

# %%
