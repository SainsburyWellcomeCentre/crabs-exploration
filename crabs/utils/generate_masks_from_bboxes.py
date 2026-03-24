"""Generate SAM2 masks from provided bboxes annotations.

Pass bounding boxes as prompts to SAM2, predict masks in those regions,
and save as an ID-encoded zarr store.

Expected data directory structure::

    <data_dir>/
        frames/                                 # image files (*.png)
        annotations/                            # contains COCO-format JSON
            VIA_JSON_combined_coco_gen.json

Output masks are written to:
    <data_dir>/annotations/masks_<timestamp>.zarr

The bbox format for SAM is a (N, 4) float array of
[x1, y1, x2, y2] in pixel coordinates, where x = column (width axis),
y = row (height axis), 1 is the xmin, ymin corner and 2 is the
xmax ymax corner.

Usage (dependencies are auto-installed via uv):
    uv run generate_masks_from_bboxes.py /path/to/data_dir
    uv run generate_masks_from_bboxes.py /path/to/data_dir --batch-size 8
    uv run generate_masks_from_bboxes.py /path/to/data_dir \
        --model facebook/sam2.1-hiera-large
"""

# /// script
# requires-python = ">=3.11"
# dependencies = [
#   "Pillow",
#   "zarr",
#   "ethology",
#   "sam2 @ git+https://github.com/facebookresearch/sam2.git",
# ]
# ///

import argparse
import sys
from datetime import datetime
from pathlib import Path

import numpy as np
import torch
import zarr
from ethology.io.annotations import load_bboxes
from PIL import Image
from sam2.sam2_image_predictor import SAM2ImagePredictor


class ImageArrayLazy:
    """A lazy array for images in a list."""

    def __init__(self, img_paths):
        """Initialise with a list of image file paths."""
        self.img_paths = sorted(img_paths)
        # add image shape, assuming all have same as
        # first sample
        sample = np.array(Image.open(img_paths[0]))  # H, W, C
        self.img_h, self.img_w, self.img_c = sample.shape

    def __len__(self):
        """Return the number of images."""
        return len(self.img_paths)

    def __getitem__(self, idx):
        """Load and return image at the given index as a numpy array."""
        return np.array(Image.open(self.img_paths[idx]))

    @property
    def shape(self):
        """Return the shape ``(n_images, height, width, channels)``."""
        return (len(self.img_paths), self.img_h, self.img_w, self.img_c)
        # B, H, W, C


def get_bboxes_x1y1_x2y2_per_frame(ds_bboxes):
    """Build a dict mapping frame index to (N, 4) bbox arrays.

    Converts centre+shape format to [x1, y1, x2, y2] corners.
    """
    x1y1 = ds_bboxes.position - ds_bboxes.shape / 2
    x2y2 = ds_bboxes.position + ds_bboxes.shape / 2

    return {
        idx: np.c_[
            x1y1.sel(image_id=idx).dropna(dim="id", how="all").values.T,
            x2y2.sel(image_id=idx).dropna(dim="id", how="all").values.T,
        ]
        for idx in range(len(ds_bboxes.image_id))
    }


def create_mask_zarr(
    path_to_zarr,
    zarr_array_shape,
    metadata_dict=None,
):
    """Create a zarr store for ID-encoded masks and write metadata."""
    n_images = zarr_array_shape[0]
    image_h = zarr_array_shape[1]
    image_w = zarr_array_shape[2]

    mask_zarr = zarr.open(
        path_to_zarr,
        mode="w",
        shape=(n_images, image_h, image_w),
        dtype="bool",
        fill_value=False,
        chunks=(1, image_h, image_w),
    )

    if metadata_dict is not None:
        mask_zarr.attrs.update(metadata_dict)
    return mask_zarr


def predict_masks_across_images(
    model,
    image_array,
    bbox_prompts_per_frame_idx,
    output_zarr,
    img_batch_size=4,
):
    """Predict masks in batches and write to zarr store."""
    n_frames = output_zarr.shape[0]  # len(ds_bboxes.image_id)
    img_h, img_w = output_zarr.shape[
        1:3
    ]  # ds_bboxes.attrs["image_array"].shape[1:3]

    for idx in range(0, n_frames, img_batch_size):
        actual_batch_size = min(img_batch_size, n_frames - idx)

        # Integer array (B, H, W):
        # 0 = background, 1+ = object instance IDs
        id_mask_batch = np.zeros(
            (actual_batch_size, img_h, img_w), dtype=np.int16
        )

        # Compute embeddings for image batch
        image_batch = [image_array[idx + i] for i in range(actual_batch_size)]
        model.set_image_batch(image_batch)

        # List of (N_i, 4) arrays, one per image
        boxes_batch = [
            bbox_prompts_per_frame_idx[f_i]
            for f_i in range(idx, idx + actual_batch_size)
        ]

        # Predict batch of masks — list of (N, 1, H, W) arrays
        masks_batch, _scores_batch, _ = model.predict_batch(
            box_batch=boxes_batch,
            multimask_output=False,
        )

        # Convert boolean masks to ID-encoded masks
        # (higher ID wins in overlap)
        for idx_rel_batch in range(actual_batch_size):
            masks_one_frame = masks_batch[idx_rel_batch].squeeze(
                axis=1
            )  # (N, H, W)

            n_objects = masks_one_frame.shape[0]
            obj_ids = np.arange(1, n_objects + 1, dtype=np.int16)[
                :, None, None
            ]
            id_mask_batch[idx_rel_batch] = (masks_one_frame * obj_ids).max(
                axis=0
            )

            print(
                f"Frame {idx + idx_rel_batch}: "
                f"{n_objects} masks / "
                f"{boxes_batch[idx_rel_batch].shape[0]}"
                " boxes"
            )

        # Save id_mask to zarr store
        output_zarr[idx : idx + actual_batch_size] = id_mask_batch

        # Mark frames as annotated in zarr store attributes
        annotated = set(output_zarr.attrs.get("annotated_frames", []))
        annotated.update(range(idx, idx + actual_batch_size))
        output_zarr.attrs["annotated_frames"] = sorted(annotated)


def main(args):
    """Generate masks from bbox annotations using SAM2."""
    # Parse CLI args
    data_dir = args.data_dir
    images_dir = data_dir / "frames"
    annotations_dir = data_dir / "annotations"
    annotations_file = annotations_dir / "VIA_JSON_combined_coco_gen.json"
    batch_size = args.batch_size
    model_id = args.model

    # Set device
    if torch.cuda.is_available():
        device = "cuda"
    elif torch.backends.mps.is_available():
        device = "mps"
    else:
        device = "cpu"
    print(f"Using device: {device}")

    # Create output zarr store
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_masks_zarr = annotations_dir / f"masks_{timestamp}.zarr"
    output_masks_zarr.mkdir(parents=True, exist_ok=True)

    # Read groundtruth as ethology annotation dataset
    ds_bboxes = load_bboxes.from_files(
        annotations_file,
        format="COCO",
        images_dirs=images_dir,
    )

    # Load ground truth images as lazy array
    png_files = sorted(ds_bboxes.attrs["images_directories"].glob("*.png"))
    image_array = ImageArrayLazy(png_files)
    # ds_bboxes.attrs["image_array"] = image_array

    # Load SAM2 predictor on device
    image_predictor = SAM2ImagePredictor.from_pretrained(
        model_id=model_id,
        device=device,
    )

    # Create zarr store for output ID-encoded masks
    n_images, image_h, image_w = image_array.shape[:3]  # (T, H, W)
    metadata_dict = {
        "timestamp": timestamp,
        "sam2_model": model_id,
        "source_images_dir": str(images_dir),
        "annotations_file": str(annotations_file),
        "annotations_format": "COCO",
        "n_images": n_images,
        "image_shape": [image_h, image_w],
        "batch_size": batch_size,
        "multimask_output": False,
        "prompt_type": "bounding_box",
        "mask_encoding": "instance_id",
        "background_label": 0,
        "id_offset": 1,
        "device": device,
    }
    mask_zarr = create_mask_zarr(
        output_masks_zarr,
        (n_images, image_h, image_w),
        metadata_dict=metadata_dict,
    )

    # Compute bbox prompts per frame,
    # as dict mapping frame index to
    # (N, 4) bbox arrays in [x1, y1, x2, y2] format
    map_frame_idx_to_boxes = get_bboxes_x1y1_x2y2_per_frame(ds_bboxes)

    # Predict masks in image batches
    # and write to zarr store in OCTRON-format
    predict_masks_across_images(
        image_predictor,
        image_array,
        map_frame_idx_to_boxes,
        mask_zarr,
        batch_size,
    )

    print(f"Saved ID-encoded mask zarr to {output_masks_zarr}")


def parse_args(list_args: list[str]) -> argparse.Namespace:
    """Parse CLI args."""
    parser = argparse.ArgumentParser(
        description="Run SAM2 on bounding box annotations to generate masks",
    )
    parser.add_argument(
        "data_dir",
        type=Path,
        help=(
            "Root data directory. Must contain a 'frames/' "
            "sub-directory with *.png images and an "
            "'annotations/' sub-directory with a "
            "VIA_JSON_combined_coco_gen.json file."
        ),
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=4,
        help="Number of images per batch (default: 4).",
    )
    parser.add_argument(
        "--model",
        type=str,
        default="facebook/sam2.1-hiera-base-plus",
        help=(
            "HuggingFace model ID for SAM2 "
            "(default: facebook/sam2.1-hiera-base-plus)."
        ),
    )
    return parser.parse_args(list_args)


if __name__ == "__main__":
    args = parse_args(sys.argv[1:])
    main(args)
