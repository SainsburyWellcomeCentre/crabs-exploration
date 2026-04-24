"""SAM3 for detecting burrows.

Standalone script: pass bounding boxes as prompts to SAM3 via OCTRON,
detect all similar instances with text and visual exemplars, then export
YOLO detection training data — without using the GUI.

BBox format: (N, 4) float32 array of [x1, y1, x2, y2] in pixel coordinates
  x = column (width axis), y = row (height axis)

Dependencies are auto-installed if the script is executed via uv run
"""
# /// script
# requires-python = ">=3.11"
# dependencies = [
#   "Pillow",
#   "octron[all] @ git+https://github.com/OCTRON-tracking/OCTRON-GUI.git",
#   "ethology",
# ]
# ///


# %%
from pathlib import Path

import numpy as np
import zarr
from ethology.io.annotations import load_bboxes
from octron.sam_octron.helpers.build_sam2_octron import build_sam2_octron
from octron.sam_octron.helpers.sam2_zarr import (
    create_image_zarr,
    get_annotated_frames,
    load_image_zarr,
    mark_frames_annotated,
)
from octron.yolo_octron.yolo_octron import YOLO_octron
from PIL import Image

# %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
# Input data

# Predictor
SAM2_CKPT_PATH = "/Users/sofia/swc/project_octron/OCTRON-GUI/octron/sam_octron/checkpoints/sam2.1_hiera_base_plus.pt"
SAM2_CONFIG_PATH = "/Users/sofia/swc/project_octron/OCTRON-GUI/octron/sam_octron/configs/sam2.1/sam2.1_hiera_b+.yaml"

# September groundtruth data
DATA_DIR = Path("/Users/sofia/swc/CrabLabels/sep2023-full")
IMAGES_DIR = DATA_DIR / "frames"
ANNOTATIONS_FILE = DATA_DIR / "annotations" / "VIA_JSON_combined_coco_gen.json"

# Prediction params
TEXT_PROMPT = "crab"  # set to None to not use
CONF_THRESHOLD = 0.5

# Output dir
OUTPUT_DIR = Path("./output")  # root output folder

# %%%%%%%%%%%%%%%%%%%%%%%%%%
# Helpers


class ImageArrayLazy:
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
        return (
            len(self.img_paths),
            self.img_h,
            self.img_w,
            self.img_c,
        )  # B, H, W, C


# %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
# Create output masks dir
# data_hash = DATA_DIR.name
# annotation_dir = CRAB_MASKS_DIR / data_hash
# annotation_dir.mkdir(parents=True, exist_ok=True)


# %%%%%%%%%%%%%%%%%%%%%%
# Read groundtruth as ethology annotation dataset
ds_bboxes = load_bboxes.from_files(
    ANNOTATIONS_FILE,
    format="COCO",
    images_dirs=IMAGES_DIR,
)

# %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
# Load ground truth images as lazy array

# build lazy image array
png_files = sorted(ds_bboxes.attrs["images_directories"].glob("*.png"))
image_array = ImageArrayLazy(png_files)
print(image_array.shape)

# add as attribute?
ds_bboxes.attrs["image_array"] = image_array


# %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
# Load predictor
predictor, device = build_sam2_octron(
    config_file_path=SAM2_CONFIG_PATH,
    ckpt_path=SAM2_CKPT_PATH,
)

# octron flag to track if init_state() has been called
predictor.is_initialized = False


# %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
# Initialise zarr cache for predictor
# When the predictor accesses a frame for the first time
# it loads + resizes it from disk and writes it into the zarr;
# on subsequent accesses it reads from the cache instead of re-decoding.
# This keeps memory usage flat for long videos.

sam_img_size = predictor.image_size  # 1024 for SAM2
store = zarr.storage.MemoryStore()
zarr_array = zarr.create_array(
    store=store,
    shape=(image_array.shape[0], 3, sam_img_size, sam_img_size),
    dtype="float16",
    fill_value=np.nan,
)

# initialise inference state with data
# (connects predictor to image data and cache)
predictor.init_state(video_data=image_array, zarr_store=zarr_array)
predictor.is_initialized = True

# %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
# Define per-frame exemplar bboxes from groundtruth

ds_bboxes = load_bboxes.from_files(
    ANNOTATIONS_FILE,
    format="COCO",
    images_dirs=IMAGES_DIR,
)

# corner 1 is min x, min y
# corner 2 is max x, max y -- check!
x1y1 = ds_bboxes.position - ds_bboxes.shape / 2
x2y2 = ds_bboxes.position + ds_bboxes.shape / 2

# Each key is a frame index; value is an (N, 4) float32 array [x1, y1, x2, y2].

map_frame_idx_to_boxes = {
    idx: np.c_[
        x1y1.sel(image_id=idx).dropna(dim="id", how="all").values.T,
        x2y2.sel(image_id=idx).dropna(dim="id", how="all").values.T,
    ]
    for idx in range(len(ds_bboxes.image_id))
}

# %%%%%%%%%%%%%%%%%%%%%%%%%%%
# Initialise zarr store for output masks
data_hash = DATA_DIR.name
annotation_dir = OUTPUT_DIR / data_hash
annotation_dir.mkdir(parents=True, exist_ok=True)

mask_zarr = create_image_zarr(
    zarr_path=annotation_dir / f"{TEXT_PROMPT} masks.zarr",
    num_frames=ds_bboxes.attrs["image_array"].shape[0],
    image_height=ds_bboxes.attrs["image_array"].shape[1],
    image_width=ds_bboxes.attrs["image_array"].shape[2],
    fill_value=-1,
    dtype="int16",
    video_hash_abbrev=data_hash,
)

# %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
# Run detection on all frames using prompts
# and save results in OCTRON semantic zarr format.

for frame_idx, bboxes in map_frame_idx_to_boxes.items():
    # Detect in current frame
    pred_masks, pred_scores, _class_idcs = predictor.detect(
        frame_idx=frame_idx,
        text=TEXT_PROMPT,
        bboxes=bboxes.tolist(),
        conf_threshold=CONF_THRESHOLD,
    )
    if pred_masks is None or pred_masks.shape[0] == 0:
        print(f"Frame {frame_idx}: no detections above threshold")
        continue
    print(f"Frame {frame_idx}: detected {pred_masks.shape[0]} objects")

    # Format result
    # Inside the zarr (shape T×H×W, dtype int16, fill -1):
    #   pixel value  0  = background
    #   pixel value  1  = highest-confidence detection
    #   pixel value  2  = second-highest, etc.
    # Mirrors _handle_semantic_box_detection() in sam2_layer_callback.py
    # (that file handles all predictor variants including SAM3 Mode B).

    id_mask = np.zeros((H, W), dtype=np.int16)  # 0 = background

    # Loop in order of descending score
    sorted_indices = pred_scores.argsort(descending=True)
    for k, det_idx in enumerate(sorted_indices):
        binary = pred_masks[det_idx].cpu().numpy()
        id_mask[(binary) & (id_mask == 0)] = (
            k + 1
        )  # 1-based, higher confidence first
    # ------------

    # Save id_mask to zarr store
    mask_zarr[frame_idx] = id_mask

    # Mark frames as annotated in zarr store attributes
    mark_frames_annotated(mask_zarr, frame_idx)


print(
    f"Saved ID-encoded mask zarr to {annotation_dir / f'{TEXT_PROMPT} masks.zarr'}"
)
# %%%%%%%%%%%%%%%%%%%%%%%
# Review in napari

# %%%%%%%%%%%%%%%%%%%%%%%
# Generate YOLO detection training data.
#
# collect_labels() normally loads video via FastVideoReader + hashes the file,
# which doesn't work for a directory of PNGs.  Instead we populate
# yolo.label_dict manually with the numpy array and zarr masks already in memory.

# %%
yolo = YOLO_octron(project_path=OUTPUT_DIR, clean_training_dir=True)
yolo.train_mode = "detect"

# %%
# Build label_dict manually — same structure that collect_labels() returns.
# See octron/yolo_octron/helpers/training.py::collect_labels()
#
# Structure:
#   label_dict[subfolder_path] = {
#       label_id: {label, frames, masks, color, original_id},
#       'video': indexable array  (video_data[frame_id] -> H×W×3 uint8),
#       'video_file_path': Path,
#   }

loaded_masks, status = load_image_zarr(
    zarr_path=annotation_dir / f"{TEXT_PROMPT} masks.zarr",
    num_frames=T,
    image_height=H,
    image_width=W,
    num_ch=None,
    verbose=True,
)
assert status, "Failed to load mask zarr"

annotated_frames = get_annotated_frames(loaded_masks)
print(f"Found {len(annotated_frames)} annotated frames")

yolo.label_dict = {
    annotation_dir.as_posix(): {
        0: {
            "label": TEXT_PROMPT,
            "original_id": 0,
            "frames": annotated_frames,
            "masks": [loaded_masks],
            "color": [1.0, 0.0, 0.0, 1.0],
        },
        "video": video_data,  # numpy array, indexable by frame
        "video_file_path": VIDEO_PATH,
    }
}

# %%
# Step 2: extract bboxes from id-encoded masks
for _ in yolo.prepare_bboxes():
    pass  # consumes the generator (prints progress via tqdm)

# Step 3: train/val/test split
yolo.prepare_split(
    training_fraction=0.7, validation_fraction=0.15, verbose=True
)

# Step 4: export images + YOLO .txt label files
for _ in yolo.create_training_data_detect(verbose=True):
    pass

# Step 5: write YOLO config
yolo.write_yolo_config(train_mode="detect")

print(f"YOLO training data ready at: {yolo.data_path}")
print(f"YOLO config written to: {yolo.config_path}")
