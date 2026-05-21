"""Run SAM3 image inference on burrow frames using bbox/point prompts.

Follows the official SAM3 image predictor example:
https://github.com/facebookresearch/sam3/blob/main/examples/sam3_image_predictor_example.ipynb
and https://github.com/facebookresearch/sam3#basic-usage  


Prompt data is produced upstream (one CSV per video, grouped by ``group_id``)
with columns:
  prompt_point_x, prompt_point_y,
  prompt_bbox_xmin, prompt_bbox_ymin, prompt_bbox_xmax, prompt_bbox_ymax

Only the bbox columns are used here. With ``PROMPT_TYPE = "point"`` the point
prompts are *derived* from the bboxes (darkest pixel near the bbox centre),
not read from the CSV ``prompt_point_*`` columns.
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
import zarr
from PIL import Image
from sam3 import build_sam3_image_model
from sam3.model.box_ops import box_xywh_to_cxcywh
from sam3.model.sam3_image_processor import Sam3Processor
from sam3.visualization_utils import (
    draw_box_on_image,
    normalize_bbox,
)


# %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
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
CONF_THRESHOLD = 0.3

# Geometric prompt type: "bounding_box" or "point".
# "point" uses the darkest-pixel
# points *derived* from the CSV bboxes (see derive_points_from_bboxes).
PROMPT_TYPE = "bounding_box"

# Fraction of each bbox (centred) searched for the darkest pixel when
# deriving a point prompt from a bbox.
CENTRAL_FRACTION = 0.5

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


def add_point_prompt(processor, state, point_xy, label=True):
    """Add a single point prompt and run inference, returning the updated state.

    ``point_xy`` is an ``(x, y)`` pair normalized to ``[0, 1]``. Mirrors
    ``Sam3Processor.add_geometric_prompt`` but appends a point to the geometric
    prompt instead of a box. Relies on SAM3 internals (``_get_dummy_prompt`` /
    ``_forward_grounding``) as the processor exposes no public point method.
    """
    if "backbone_out" not in state:
        raise ValueError("call processor.set_image before adding a prompt")
    if "language_features" not in state["backbone_out"]:
        # no text prompt yet: fall back to a dummy "visual" text prompt so the
        # model relies only on the geometric prompt
        dummy_text = processor.model.backbone.forward_text(
            ["visual"], device=processor.device
        )
        state["backbone_out"].update(dummy_text)
    if "geometric_prompt" not in state:
        state["geometric_prompt"] = processor.model._get_dummy_prompt()

    # points: (n_points, batch, 2); labels: (n_points, batch); mask: (batch, n_points)
    pts = torch.tensor(
        point_xy, device=processor.device, dtype=torch.float32
    ).view(1, 1, 2)
    lbl = torch.tensor(
        [label], device=processor.device, dtype=torch.bool
    ).view(1, 1)
    msk = torch.zeros(1, 1, dtype=torch.bool, device=processor.device)
    state["geometric_prompt"].append_points(pts, lbl, msk)

    return processor._forward_grounding(state)


def bbox_to_darkest_point(image_hwc, bbox_xyxy, central_fraction):
    """Return the darkest pixel ``(x, y)`` within the centre of a bbox.

    The search is restricted to a centred sub-region of the bbox spanning
    ``central_fraction`` of its width and height. ``bbox_xyxy`` is in pixel
    ``(xmin, ymin, xmax, ymax)`` coordinates; the returned point is in pixel
    ``(x, y)`` coordinates of the full image.
    """
    x1, y1, x2, y2 = bbox_xyxy
    cx, cy = (x1 + x2) / 2.0, (y1 + y2) / 2.0
    sub_w = (x2 - x1) * central_fraction
    sub_h = (y2 - y1) * central_fraction

    img_h, img_w = image_hwc.shape[:2]
    sx1 = int(round(np.clip(cx - sub_w / 2.0, 0, img_w - 1)))
    sx2 = int(round(np.clip(cx + sub_w / 2.0, 0, img_w)))
    sy1 = int(round(np.clip(cy - sub_h / 2.0, 0, img_h - 1)))
    sy2 = int(round(np.clip(cy + sub_h / 2.0, 0, img_h)))

    # degenerate crop: fall back to the bbox centre
    if sx2 <= sx1 or sy2 <= sy1:
        return (float(cx), float(cy))

    # grayscale crop -> darkest pixel
    gray = np.asarray(image_hwc).astype(np.float32)
    if gray.ndim == 3:
        gray = gray.mean(axis=2)
    crop = gray[sy1:sy2, sx1:sx2]
    dy, dx = np.unravel_index(int(np.argmin(crop)), crop.shape)
    return (float(sx1 + dx), float(sy1 + dy))


def derive_points_from_bboxes(image_hwc, bboxes_xyxy, central_fraction):
    """Derive one darkest-pixel point per bbox; returns an ``(N, 2)`` array."""
    if len(bboxes_xyxy) == 0:
        return np.empty((0, 2), dtype=np.float32)
    return np.array(
        [
            bbox_to_darkest_point(image_hwc, bbox, central_fraction)
            for bbox in bboxes_xyxy
        ],
        dtype=np.float32,
    )


def add_geometric_prompts(processor, state, boxes_cxcywh_norm, labels=None):
    """Add multiple box prompts at once and run inference a single time.

    ``boxes_cxcywh_norm`` is an ``(N, 4)`` array of cxcywh boxes normalized to
    ``[0, 1]``. Mirrors ``Sam3Processor.add_geometric_prompt`` but appends the
    whole batch to the geometric prompt before one ``_forward_grounding`` call,
    avoiding the N-1 redundant grounding passes of a per-box loop.

    The official docs use the public per-box loop because 
    add_geometric_prompt has no batched form. 
    """
    if "backbone_out" not in state:
        raise ValueError("call processor.set_image before adding a prompt")
    if "language_features" not in state["backbone_out"]:
        # no text prompt: fall back to a dummy "visual" text prompt
        dummy_text = processor.model.backbone.forward_text(
            ["visual"], device=processor.device
        )
        state["backbone_out"].update(dummy_text)
    if "geometric_prompt" not in state:
        state["geometric_prompt"] = processor.model._get_dummy_prompt()

    # boxes: (n_boxes, batch, 4); labels: (n_boxes, batch)
    boxes = torch.tensor(
        boxes_cxcywh_norm, device=processor.device, dtype=torch.float32
    ).view(-1, 1, 4)
    n = boxes.shape[0]
    if labels is None:
        labels = torch.ones(n, 1, dtype=torch.bool, device=processor.device)
    else:
        labels = torch.as_tensor(
            labels, device=processor.device, dtype=torch.bool
        ).view(n, 1)
    state["geometric_prompt"].append_boxes(boxes, labels)

    return processor._forward_grounding(state)


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
    "prompt_type": PROMPT_TYPE,
    "central_fraction": CENTRAL_FRACTION,
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
# Selection is always on the bboxes: click a box to toggle it
# (red = selected/kept, lime = excluded). In "point" mode the derived
# point prompts are additionally drawn as `x` markers for reference.

import matplotlib.patches as patches

select_frame_idx = 0
if flag_using_date_prompts:
    group_str_for_prompts = list_date_per_img[select_frame_idx]
else:
    group_str_for_prompts = list_video_per_img[select_frame_idx]

# master prompt array: always the CSV bboxes. `selected` is a mask over
# its indices; in "point" mode the derived points map 1:1 to these bboxes.
bboxes_select = bboxes_xyxy_per_video[group_str_for_prompts]
prompts_select = bboxes_select
selected = np.zeros(len(bboxes_select), dtype=bool)  # start all unselected

# derived darkest-pixel points, only used as an overlay in "point" mode
if PROMPT_TYPE == "point":
    derived_points = derive_points_from_bboxes(
        image_array[select_frame_idx], bboxes_select, CENTRAL_FRACTION
    )

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

# in "point" mode, an `x` marker per derived point, also index-aligned
# with `selected` (reference only — clicks still toggle the boxes)
point_artists = []
if PROMPT_TYPE == "point":
    for x, y in derived_points:
        (pt,) = ax.plot(
            x, y,
            marker="x",
            markersize=14,
            markeredgecolor="lime",
        )
        point_artists.append(pt)

def _set_color(i):
    color = "red" if selected[i] else "lime"
    artists[i].set_edgecolor(color)
    if PROMPT_TYPE == "point":
        point_artists[i].set_markeredgecolor(color)

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
# filter the bboxes (derived points map 1:1 to bboxes, so the mask applies
# regardless of PROMPT_TYPE)
bboxes_xyxy_per_video[group_str_for_prompts] = bboxes_select[selected]
print(f"{group_str_for_prompts}: kept {selected.sum()} prompts")


# %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
# Run inference on every frame and write ID-encoded masks to zarr

processed_frames = []

for frame_idx in [0]: #range(len(image_array)):
    # with torch.autocast("cuda", dtype=torch.bfloat16):

    # Get corresponding video
    if flag_using_date_prompts:
        video_str = list_date_per_img[frame_idx]
    else:
        video_str = list_video_per_img[frame_idx]

    # Get bbox prompts for that video (the master prompt store)
    prompts = bboxes_xyxy_per_video.get(video_str)
    if prompts is None or len(prompts) == 0:
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

    # Add geometric prompts (boxes or points)
    if PROMPT_TYPE == "bounding_box":
        # bbox exemplars: xyxy (pixels) -> xywh -> cxcywh -> normalized
        boxes_xywh = prompts.astype(np.float32).copy()
        boxes_xywh[:, 2] -= boxes_xywh[:, 0]  # w = xmax - xmin
        boxes_xywh[:, 3] -= boxes_xywh[:, 1]  # h = ymax - ymin
        boxes_cxcywh = box_xywh_to_cxcywh(torch.tensor(boxes_xywh).view(-1, 4))
        norm_boxes_cxcywh = normalize_bbox(
            boxes_cxcywh, width, height
        ).tolist()
        # for box in norm_boxes_cxcywh:
        #     inference_state = processor.add_geometric_prompt(
        #         state=inference_state, box=box, label=True
        #     )
        inference_state = add_geometric_prompts(
            processor, inference_state, norm_boxes_cxcywh
        )
    else:
        # derive a darkest-pixel point per bbox from *this* frame's image,
        # then xy (pixels) -> normalized [0, 1]
        points_xy = derive_points_from_bboxes(
            image_array[frame_idx], prompts, CENTRAL_FRACTION
        )
        norm_points_xy = points_xy / np.array(
            [width, height], dtype=np.float32
        )
        for px, py in norm_points_xy:
            inference_state = add_point_prompt(
                processor, inference_state, (float(px), float(py)), label=True
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

    # Compute id-encoded mask
    # n_masks may be different from n_objects because:
    # - SAM3 can return a mask that is all False
    # - when computing the id mask, if masks overlap we take the one with 
    #   higher ID. So completely overlapping masks disappear.
    obj_ids = np.arange(1, n_objects + 1, dtype=np.int16)[:, None, None]
    id_mask = (masks.astype(bool) * obj_ids).max(axis=0)

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

# always draw the source bboxes
image_with_boxes = image
for x1, y1, x2, y2 in bboxes_xyxy_per_video[video_str]:
    image_with_boxes = draw_box_on_image(
        image_with_boxes, [x1, y1, x2 - x1, y2 - y1], (0, 255, 0)
    )

plt.figure()
plt.imshow(image_with_boxes)
if PROMPT_TYPE == "point":
    # overlay the darkest-pixel points derived for this frame
    pts = derive_points_from_bboxes(
        image_array[frame_idx],
        bboxes_xyxy_per_video[video_str],
        CENTRAL_FRACTION,
    )
    plt.scatter(
        pts[:, 0], pts[:, 1],
        c="lime", marker="*", s=120, edgecolors="k",
    )
plt.axis("off")
plt.title(f"frame {frame_idx} ({video_str}) - prompt {PROMPT_TYPE}")
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

# n_masks may be different from n_objects because:

# %%
