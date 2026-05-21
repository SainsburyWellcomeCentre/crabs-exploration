"""Run SAM3 image inference on burrow frames using bbox prompts.

Ultralytics implementation (matches OCTRON). The grounding head is called
directly, bypassing ``Sam3Processor`` postprocessing.

Follows the official SAM3 image predictor example:
https://github.com/facebookresearch/sam3/blob/main/examples/sam3_image_predictor_example.ipynb
and https://github.com/facebookresearch/sam3#basic-usage


Prompt data is produced upstream (one CSV per video, grouped by ``group_id``)
with columns:
  prompt_point_x, prompt_point_y,
  prompt_bbox_xmin, prompt_bbox_ymin, prompt_bbox_xmax, prompt_bbox_ymax

Only the bbox columns are used here.
"""
# /// script
# requires-python = ">=3.11"
# dependencies = [
#   "numpy",
#   "pandas",
#   "matplotlib",
#   "Pillow",
#   "zarr",
#   "torch>=2.5.1",
#   "torchvision>=0.20.1",
#   "ultralytics",
#   "ipympl",
# ]
# ///


# %%
# uv venv .venv_ultralytics --python=3.11
# source .venv_ultralytics/bin/activate
# uv sync --script notebook_ultralytics_sam3.py

# %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
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
import torch.nn.functional as F
import zarr
from PIL import Image
from ultralytics.models.sam.build_sam3 import build_sam3_image_model
from ultralytics.models.sam.sam3.geometry_encoders import Prompt
from ultralytics.utils import ops


# %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
# Input data
# - images_dir: directory of PNG frames
# - prompt_coords_dir: directory of per-video prompt CSVs

images_dir = "/home/sminano/swc/project_crabs/burrow_mean_image_slurm_3014447"
prompt_coords_dir = "/home/sminano/swc/project_crabs/burrow_prompts_per_day_20260423_143244"
#"/home/sminano/swc/project_crabs/burrow_prompts_slurm_3012602/coords_20260519_105922"

# Select whether to use video prompts or date prompts
flag_using_date_prompts = True

# Prediction params
TEXT_PROMPT = "burrow"  # set to None to skip the text prompt
CONF_THRESHOLD = 0.5

# Output dir for masks
# TODO: add timestamp
OUTPUT_DIR = Path("/home/sminano/swc/project_crabs/crabs-exploration/output_burrows_sam3")

# use only the top N prompt boxes per video (or date)
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
# Build SAM3 detector — ultralytics implementation (matches OCTRON)

device = torch.device(
    "cuda" if torch.cuda.is_available()
    else "mps" if torch.backends.mps.is_available()
    else "cpu"
)

# to avoid bfloat16 and float mismatch
if torch.cuda.is_available():
    torch.autocast("cuda", dtype=torch.bfloat16).__enter__()

# %%

SAM3_CKPT = "/home/sminano/swc/project_octron/OCTRON-GUI/octron/sam_octron/checkpoints/sam3.pt"  # adjust to your path
detector = build_sam3_image_model(SAM3_CKPT)
detector = detector.to(device).eval()

# SAM3 geometry — matches octron/sam_octron/helpers/sam3_octron.py
SAM3_IMAGE_SIZE = 1008
SAM3_IMG_MEAN = torch.tensor([0.5, 0.5, 0.5])[:, None, None]
SAM3_IMG_STD = torch.tensor([0.5, 0.5, 0.5])[:, None, None]


# %%%%%%%%%%
# Drop-in replica of SAM3_semantic_octron.detect()

def preprocess_image(img_hwc_uint8, device):
    """HWC uint8 -> (1, 3, 1008, 1008) normalized — replicates OctoZarr/SAM3."""
    t = torch.from_numpy(np.asarray(img_hwc_uint8)).permute(2, 0, 1).float() / 255.0
    t = F.interpolate(
        t[None], size=(SAM3_IMAGE_SIZE, SAM3_IMAGE_SIZE),
        mode="bilinear", align_corners=False,
    )[0]
    t = (t - SAM3_IMG_MEAN) / SAM3_IMG_STD
    return t[None].to(device)


@torch.inference_mode()
def octron_detect(detector, image_hwc, bboxes_xyxy, video_w, video_h,
                  text=None, conf_threshold=0.5, device="cpu"):
    """Raw grounding queries above threshold — NO NMS (matches OCTRON)."""
    image = preprocess_image(image_hwc, device).float()
    features = detector.backbone.forward_image(image)

    # Text classes
    text_batch = [text] if isinstance(text, str) else (text or ["visual"])
    nc = len(text_batch)
    if detector.names != text_batch:
        detector.set_classes(text=text_batch)

    # Geometric (box) prompts
    geometric_prompt = Prompt(
        box_embeddings=torch.zeros(0, nc, 4, device=device),
        box_mask=torch.zeros(nc, 0, device=device, dtype=torch.bool),
    )
    if bboxes_xyxy is not None and len(bboxes_xyxy) > 0:
        b = torch.as_tensor(bboxes_xyxy, dtype=torch.float32, device=device)
        if b.ndim == 1:
            b = b[None]
        b_xywh = ops.xyxy2xywh(b)                 # -> cx, cy, w, h
        b_xywh[:, 0::2] /= video_w
        b_xywh[:, 1::2] /= video_h
        labels_t = torch.ones(b_xywh.shape[0], dtype=torch.int32, device=device)
        for i in range(len(b_xywh)):
            geometric_prompt.append_boxes(
                b_xywh[i].view(1, 1, 4), labels_t[i].view(1, 1)
            )

    # Run the grounding head directly (bypasses Sam3Processor postprocessing)
    text_ids = torch.arange(nc, device=device, dtype=torch.long)
    outputs = detector.forward_grounding(
        backbone_out=features,
        text_ids=text_ids,
        geometric_prompt=geometric_prompt,
    )

    pred_masks = outputs["pred_masks"]
    pred_scores = outputs["pred_logits"].sigmoid()
    if "presence_logit_dec" in outputs:
        presence = outputs["presence_logit_dec"].sigmoid().unsqueeze(1)
        pred_scores = (pred_scores * presence).squeeze(-1)
    else:
        pred_scores = pred_scores.squeeze(-1)

    # Keep every query above threshold — NO non-max suppression
    keep = pred_scores > conf_threshold
    pred_masks, pred_scores = pred_masks[keep], pred_scores[keep]
    if pred_masks.shape[0] == 0:
        return None, None

    pred_masks = F.interpolate(
        pred_masks.float()[None], size=(video_h, video_w), mode="bilinear",
    )[0] > 0
    return pred_masks, pred_scores


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
# Interactively select which prompts to pass to SAM3.
# Click a box to toggle it (red = selected/kept, lime = excluded).

import matplotlib.patches as patches

select_frame_idx = 0
if flag_using_date_prompts:
    group_str_for_prompts = list_date_per_img[select_frame_idx]
else:
    group_str_for_prompts = list_video_per_img[select_frame_idx]

# master prompt array: the CSV bboxes. `selected` is a mask over its indices.
bboxes_select = bboxes_xyxy_per_video[group_str_for_prompts]
prompts_select = bboxes_select
selected = np.zeros(len(bboxes_select), dtype=bool)  # start all unselected

fig, ax = plt.subplots()
ax.imshow(image_array[select_frame_idx])
ax.set_axis_off()

# one rectangle artist per bbox, index-aligned with `selected`
artists = []
for x1, y1, x2, y2 in prompts_select:
    r = patches.Rectangle(
        (x1, y1), x2 - x1, y2 - y1,
        fill=False, linewidth=2, edgecolor="lime",
    )
    ax.add_patch(r)
    artists.append(r)

def _set_color(i):
    artists[i].set_edgecolor("red" if selected[i] else "lime")

def _refresh_title():
    ax.set_title(
        f"{image_array.img_paths[select_frame_idx].name}"
        f"(prompts {group_str_for_prompts}: "
        f"{selected.sum()}/{len(selected)} selected)"
    )

def _hit_index(event):
    """Index of the bbox under the click, or None."""
    for i, (x1, y1, x2, y2) in enumerate(prompts_select):
        if x1 <= event.xdata <= x2 and y1 <= event.ydata <= y2:
            return i  # first match only — minimal handling of overlaps
    return None

def _on_click(event):
    if event.inaxes != ax:
        return
    i = _hit_index(event)
    if i is not None:
        selected[i] = not selected[i]
        _set_color(i)
    _refresh_title()
    fig.canvas.draw_idle()

_refresh_title()
fig.canvas.mpl_connect("button_press_event", _on_click)
plt.show()


# %%
# once you're happy with the selection, commit the choice back so the
# inference loop picks it up unchanged (it reads the *_per_video dict).
bboxes_xyxy_per_video[group_str_for_prompts] = bboxes_select[selected]
print(f"{group_str_for_prompts}: kept {selected.sum()} prompts")


# %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
# Run inference on every frame and write ID-encoded masks to zarr

processed_frames = []

for frame_idx in [select_frame_idx]: #range(len(image_array)):
    # with torch.autocast("cuda", dtype=torch.bfloat16):

    # Get corresponding video
    if flag_using_date_prompts:
        video_str = list_date_per_img[frame_idx]
    else:
        video_str = list_video_per_img[frame_idx]

    # Get bbox prompts for that video (the master prompt store)
    boxes_xyxy = bboxes_xyxy_per_video.get(video_str)
    if boxes_xyxy is None or len(boxes_xyxy) == 0:
        print(f"Frame {frame_idx} ({video_str}): no prompts, skipping")
        continue

    # Load image
    image_hwc = image_array[frame_idx]
    video_h, video_w = image_hwc.shape[:2]

    # Run detection (replica of SAM3_semantic_octron.detect)
    pred_masks, pred_scores = octron_detect(
        detector, image_hwc, boxes_xyxy,
        video_w=video_w, video_h=video_h,
        text=TEXT_PROMPT, conf_threshold=CONF_THRESHOLD, device=device,
    )

    # OCTRON text fallback: weak text scores -> retry box-only ("visual")
    if pred_scores is not None and pred_scores.max().item() < 0.25:
        detector.names = []
        pred_masks, pred_scores = octron_detect(
            detector, image_hwc, boxes_xyxy,
            video_w=video_w, video_h=video_h,
            text=None, conf_threshold=CONF_THRESHOLD, device=device,
        )

    # Release this frame's GPU state before the next frame
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    if pred_masks is None:
        print(f"Frame {frame_idx} ({video_str}): no detections")
        continue

    # ID-encode: highest confidence stamped first, fills only background
    masks = pred_masks.cpu().numpy()
    n_objects = masks.shape[0]
    order = pred_scores.argsort(descending=True).cpu().numpy()
    id_mask = np.zeros(masks.shape[1:], dtype=np.int16)
    for rank, idx in enumerate(order):
        id_mask[(masks[idx]) & (id_mask == 0)] = rank + 1

    mask_zarr[frame_idx] = id_mask
    processed_frames.append(frame_idx)
    mask_zarr.attrs["annotated_frames"] = processed_frames
    print(f"Frame {frame_idx} ({video_str}): {n_objects} masks")

print(f"Saved ID-encoded mask zarr to {output_masks_zarr}")


# %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
# Visualise one frame: prompt boxes + predicted masks

frame_idx = processed_frames[0]
if flag_using_date_prompts:
    video_str = list_date_per_img[frame_idx]
else:
    video_str = list_video_per_img[frame_idx]
image = Image.fromarray(image_array[frame_idx])

fig, ax = plt.subplots()
ax.imshow(image)

# always draw the source bboxes
for x1, y1, x2, y2 in bboxes_xyxy_per_video[video_str]:
    ax.add_patch(
        patches.Rectangle(
            (x1, y1), x2 - x1, y2 - y1,
            edgecolor="lime", facecolor="none", linewidth=1.5,
        )
    )
ax.axis("off")
ax.set_title(f"frame {frame_idx} ({video_str}) - prompt boxes")
plt.show()

# Plot the ID-encoded masks read back from the zarr store
# TODO: why only 3 masks?
id_mask = mask_zarr[frame_idx]  # (H, W), 0 = background
masked = np.ma.masked_where(id_mask == 0, id_mask)

# count number of masks
mask_ids = np.unique(id_mask)
mask_ids = mask_ids[mask_ids != 0]   # drop background
n_masks = len(mask_ids)

plt.figure()
plt.imshow(image)
plt.imshow(masked, cmap="tab10", alpha=0.5, interpolation="nearest")
plt.axis("off")
plt.title(f"{image_array.img_paths[frame_idx].stem} - {n_masks} masks")
plt.show()

# %%%%%%%%
# Postprocess?
# - remove large masks?
# - remove split masks, not blob-like masks?
# - can I do a second pass using predictions as prompts?

# %%
