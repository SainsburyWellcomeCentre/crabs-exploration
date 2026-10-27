"""Generate SAM2 masks from provided bboxes annotations.

Pass bounding boxes as prompts to SAM2, predict masks in those regions,
and save them as ID-encoded label images in a zarr store, together with
the bounding boxes and a few shape parameters per mask.

This script is a one-off tool to run on annotated frames whose mask labels
are only meaningful within a frame.

Expected data directory structure::

    <data_dir>/
        frames/                                 # image files (*.png)
        annotations/                            # contains COCO-format JSON
            VIA_JSON_combined_coco_gen.json

Output is written to an xarray dataset in a zarr store:
    <data_dir>/annotations/bboxes_and_masks_<timestamp>.zarr

with the following layout::

    ├── image_id, space, id             coordinates from ds_bboxes
    ├── position, shape, category, ...  (image_id, [space,] id)  from ethology
    ├── labels                          (image_id, img_h, img_w) uint16,
    │                                   0 = background
    ├── label_of                        (id,) uint16, label = position
    │                                   along id + 1
    ├── mask_centroid                   (image_id, space, id) float64
    └── mask_area, mask_area_to_convex_hull_area,
        mask_area_to_mask_bbox_area, ellipse_major_axis_length,
        ellipse_minor_axis_length, ellipse_major_axis_theta,
        ellipse_eccentricity            (image_id, id) float64

All variables share the ``image_id`` and ``id`` coordinates, so the mask
and ellipse parameters of a bounding box sit at the same (image_id, id)
as the box. Slots with no box, or with an empty mask, are NaN.

The mask parameters are computed with scikit-image's
``regionprops_table`` on the stored ``labels``, so an occluded crab is
measured on its visible pixels only ("smallest_wins" occlusion policy).
The ellipse is the one with the same second moments as the mask:

- ``mask_centroid``: mean (x, y) of the mask pixels, and ellipse centre.
- ``mask_area``: number of mask pixels.
- ``mask_area_to_convex_hull_area``: mask area / area of its convex hull.
- ``mask_area_to_mask_bbox_area``: mask area / area of the mask's own
  axis-aligned bounding box (not the annotated bounding box).
- ``ellipse_major_axis_length``, ``ellipse_minor_axis_length``: full axis
  lengths in pixels, not semi-axes.
- ``ellipse_major_axis_theta``: angle of the major axis from the image
  x-axis towards the y-axis, in radians, in [0, pi). The image y-axis
  points down, so a positive angle looks clockwise on screen. It can be
  passed to matplotlib's ``Ellipse(..., angle=np.degrees(theta))``
  directly on an image axis. It is arbitrary for near-circular masks
  (``ellipse_eccentricity`` close to 0).
- ``ellipse_eccentricity``: 0 for a circle, towards 1 for an elongated
  ellipse.

Usage
From the repo root:
    uv run scripts/orientation/generate_masks_from_bboxes.py /path/to/data_dir
    uv run scripts/orientation/generate_masks_from_bboxes.py \
        /path/to/data_dir --batch-size 8
    uv run scripts/orientation/generate_masks_from_bboxes.py \
        /path/to/data_dir --model facebook/sam2.1-hiera-large

Dependencies are auto-installed in an ephemeral environment via uv.
torch is installed from the CUDA 12.8 index (Linux/Windows).
"""

# /// script
# requires-python = ">=3.11"
# dependencies = [
#   "Pillow",
#   "zarr",
#   "xarray",
#   "dask",
#   "scikit-image",
#   "torch>=2.5.1",
#   "torchvision>=0.20.1",
#   "sam-2 @ git+https://github.com/facebookresearch/sam2.git",
#   "huggingface_hub",
#   "ethology>=0.1.4", # after stable ID assignment (PR 164)
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

import argparse
import sys
from collections.abc import Iterable
from datetime import datetime
from pathlib import Path
from typing import Any, cast

import dask.array as da
import numpy as np
import torch
import xarray as xr
import zarr
from PIL import Image
from skimage.measure import regionprops_table

# Map frame index to (N, 4) boxes and (N,) mask label IDs
# (label IDs are the mask label actual values)
BoxesAndLabelIDs = dict[int, tuple[np.ndarray, np.ndarray]]

# Label of the mask of the box at position k along `id` is k + LABEL_OFFSET;
# label 0 is background
LABEL_OFFSET = 1

# Map regionprops_table property to data variable
PROPS_TO_VARS = {
    "area": "mask_area",
    "solidity": "mask_area_to_convex_hull_area",
    "extent": "mask_area_to_mask_bbox_area",
    "axis_major_length": "ellipse_major_axis_length",
    "axis_minor_length": "ellipse_minor_axis_length",
    "eccentricity": "ellipse_eccentricity",
}


class ImageArrayLazy:
    """A lazy array for images in a list."""

    def __init__(self, img_paths: Iterable[Path]) -> None:
        """Initialise with a list of image file paths.

        The order of ``img_paths`` is preserved, so that index ``i``
        corresponds to the ``i``-th path in the list.
        """
        self.img_paths = list(img_paths)

        # all images must have the same size, so that the masks fit in
        # one array; check it from the file headers, before running SAM2
        sizes = {Image.open(p).size for p in self.img_paths}  # (W, H)
        if len(sizes) > 1:
            raise ValueError(f"Images have different sizes (W, H): {sizes}")

        sample = np.array(Image.open(self.img_paths[0]))  # H, W, C
        self.img_h, self.img_w, self.img_c = sample.shape

    def __len__(self) -> int:
        """Return the number of images."""
        return len(self.img_paths)

    def __getitem__(self, idx: int) -> np.ndarray:
        """Load and return image at the given index as a numpy array."""
        return np.array(Image.open(self.img_paths[idx]))

    @property
    def shape(self) -> tuple[int, int, int, int]:
        """Return the shape ``(n_images, height, width, channels)``."""
        return (len(self.img_paths), self.img_h, self.img_w, self.img_c)
        # B, H, W, C


def get_image_paths(ds_bboxes: xr.Dataset) -> list[Path]:
    """Return image file paths in the order of the ``image_id`` coordinate.

    The ``i``-th path corresponds to the ``i``-th element of
    ``ds_bboxes.image_id``.
    """
    # Build full paths to images
    images_dir = Path(ds_bboxes.attrs["images_directories"])
    img_paths = [
        images_dir / ds_bboxes.attrs["map_image_id_to_filename"][img_id]
        for img_id in ds_bboxes.image_id.values
    ]

    # Check if a file is missing
    missing = [p for p in img_paths if not p.is_file()]
    if missing:
        raise FileNotFoundError(
            f"{len(missing)} annotated images not found, e.g. {missing[0]}"
        )
    return img_paths


def get_bboxes_and_label_ids_per_frame(
    ds_bboxes: xr.Dataset,
) -> BoxesAndLabelIDs:
    """Build a dict mapping frame index to ``boxes`` and ``label_ids``.

    The frame index is the position along the ``image_id`` coordinate.

    ``boxes`` is an (N, 4) float array of [x1, y1, x2, y2] in pixel
    coordinates, as expected by SAM2: (x1, y1) is the top-left
    (xmin, ymin) corner and (x2, y2) the bottom-right (xmax, ymax)
    corner, with x = column and y = row.

    ``label_ids`` is an (N,) array with the label ID used to paint the
    mask associated to each box in the label image. The label ID is the
    box's position along the ``id`` dimension + 1 (0 is background).
    """
    # Get corners for all images
    x1y1 = ds_bboxes.position - ds_bboxes.shape / 2
    x2y2 = ds_bboxes.position + ds_bboxes.shape / 2

    # Loop thru images
    boxes_and_label_ids: BoxesAndLabelIDs = {}
    for idx in range(ds_bboxes.sizes["image_id"]):
        # (n_ids, 4), with one row per id slot, NaN for empty slots
        corners = np.c_[
            x1y1.isel(image_id=idx).transpose("id", "space").values,
            x2y2.isel(image_id=idx).transpose("id", "space").values,
        ]
        # select non-nan only
        non_nan = ~np.isnan(corners).all(axis=1)
        boxes_and_label_ids[idx] = (
            corners[non_nan],
            np.flatnonzero(non_nan).astype(np.uint16) + LABEL_OFFSET,
            # label ID = box's position along `id` (before dropping NaNs)
            # + LABEL_OFFSET
        )
    return boxes_and_label_ids


def create_dataset_store(
    path_to_zarr: str | Path,
    ds_bboxes: xr.Dataset,
    image_shape: tuple[int, int],
    metadata_dict: dict[str, Any],
) -> None:
    """Initialise output zarr store.

    Writes ds_bboxes, empty ``labels`` and ``label_of``.

    ``labels`` is an (image_id, img_h, img_w) uint16 array of ID-encoded
    masks, with 0 as background.

    ``label_of`` maps each ``id`` to the label of its mask, which is the
    box's position along ``id`` + 1.

    The initialised store is written with ``compute=False``: coordinates,
    metadata and the bbox variables go to disk, but no ``labels`` chunk does.
    Unwritten chunks read back as the fill value, 0.
    """
    # Get dimension sizes
    image_h, image_w = image_shape
    n_images, n_ids = ds_bboxes.sizes["image_id"], ds_bboxes.sizes["id"]
    # the largest label is n_ids - 1 + LABEL_OFFSET
    if n_ids - 1 + LABEL_OFFSET > np.iinfo(np.uint16).max:
        raise ValueError(
            f"{n_ids} id slots do not fit in uint16 labels "
            f"(max {np.iinfo(np.uint16).max})"
        )

    # Define dataset structure
    # zarr 3 has no stable format for numpy's fixed-width strings, so store
    # the `space` coordinate as an object. ethology >= 0.1.4
    # already returns it as object; the cast covers datasets built
    # before then or 'manually', where ["x", "y"] becomes <U1
    ds = ds_bboxes.assign_coords(space=ds_bboxes.space.astype(object))
    ds = ds.assign(
        labels=(
            ("image_id", "img_h", "img_w"),
            # lazy zeros, so to_zarr writes no data;
            # one dask chunk per frame, matching the zarr chunks
            da.zeros(
                (n_images, image_h, image_w),
                chunks=(1, image_h, image_w),
                dtype=np.uint16,
            ),
        ),
        # label k + LABEL_OFFSET is the mask of the box at position k
        # along `id`
        label_of=(
            "id",
            np.arange(n_ids, dtype=np.uint16) + LABEL_OFFSET,
        ),
    )
    ds.attrs = _attrs_for_zarr({**ds_bboxes.attrs, **metadata_dict})

    # Write dataset to zarr
    ds.to_zarr(
        path_to_zarr,
        mode="w",
        encoding={"labels": {"chunks": (1, image_h, image_w)}},
        # each frame is its own chunk ``(1, img_h, img_w)``
        compute=False,
    )


def _attrs_for_zarr(attrs: dict[str, Any]) -> dict[str, Any]:
    """Return a copy of ``attrs`` with every ``Path`` turned into ``str``.

    zarr cannot store ``Path`` attributes. Dict keys are left unchanged.
    """

    def fix(value: Any) -> Any:
        if isinstance(value, Path):
            return str(value)
        if isinstance(value, dict):
            return {k: fix(v) for k, v in value.items()}
        if isinstance(value, list | tuple):
            return [fix(v) for v in value]
        return value

    return {key: fix(value) for key, value in attrs.items()}


def predict_masks_across_images(
    model: Any,
    image_array: ImageArrayLazy,
    boxes_and_label_ids_per_frame_idx: BoxesAndLabelIDs,
    output_zarr: zarr.Array,
    img_batch_size: int = 4,
) -> None:
    """Predict masks in batches and write to zarr store.

    ``model`` is a ``sam2.sam2_image_predictor.SAM2ImagePredictor``.
    """
    # Get dimensions
    n_frames = output_zarr.shape[0]  # len(ds_bboxes.image_id)
    img_h, img_w = output_zarr.shape[
        1:3
    ]  # ds_bboxes.attrs["image_array"].shape[1:3]

    # Loop in batches
    for batch_start in range(0, n_frames, img_batch_size):
        # Get batch size
        actual_batch_size = min(img_batch_size, n_frames - batch_start)

        # Get label arrays (B, H, W)
        # label 0 = background
        label_mask_batch = np.zeros(
            (actual_batch_size, img_h, img_w),
            dtype=np.uint16,
        )

        # Get lists of (N_i, 4) boxes and (N_i,) label IDs, one per image
        frame_idcs = range(batch_start, batch_start + actual_batch_size)
        boxes_batch, label_ids_batch = zip(
            *(boxes_and_label_ids_per_frame_idx[f_i] for f_i in frame_idcs),
            strict=True,
        )

        # Only pass frames with at least one box to SAM2;
        # frames without boxes are left as background (0)
        # idcs_rel = indices within the batch
        idcs_rel_with_boxes: list[int] = []
        for i, boxes in enumerate(boxes_batch):
            if len(boxes):
                idcs_rel_with_boxes.append(i)
            else:
                print(f"Frame {batch_start + i}: no boxes, skipped")

        if idcs_rel_with_boxes:
            # Compute embeddings for images with boxes
            image_batch = [
                image_array[batch_start + i] for i in idcs_rel_with_boxes
            ]
            model.set_image_batch(image_batch)

            # List of (N_i, 4) arrays, one per image with boxes
            box_batch = [boxes_batch[i] for i in idcs_rel_with_boxes]

            # Predict batch of masks — list of (N, 1, H, W) arrays,
            # it has one per mask image for each frame **with boxes**
            masks_batch, _scores_batch, _ = model.predict_batch(
                box_batch=box_batch,
                multimask_output=False,
            )

            # Convert boolean masks to ID-encoded masks
            # idcs_rel = indices within the batch
            for idx_rel_batch, masks in zip(
                idcs_rel_with_boxes,
                masks_batch,
                strict=True,
            ):
                # SAM2 squeezes the box axis when N == 1, returning
                # (1, H, W) instead of (N, 1, H, W); we use reshape
                # to handle both
                masks_one_frame = masks.reshape(-1, img_h, img_w)  # (N, H, W)
                n_objects = masks_one_frame.shape[0]

                # Add results to batch array (B,H,W)
                label_mask_batch[idx_rel_batch] = flatten_masks_smallest_wins(
                    masks_one_frame, label_ids_batch[idx_rel_batch]
                )

                # log
                print(
                    f"Frame {batch_start + idx_rel_batch}: "
                    f"{n_objects} masks / "
                    f"{boxes_batch[idx_rel_batch].shape[0]}"
                    " boxes"
                )

        # Save id_mask to zarr store
        output_zarr[batch_start : batch_start + actual_batch_size] = (
            label_mask_batch
        )

        # Mark frames as annotated in zarr store attributes
        annotated = set(
            cast(
                list[int],
                output_zarr.attrs.get("annotated_frames", []),
            )
        )
        annotated.update(range(batch_start, batch_start + actual_batch_size))
        output_zarr.attrs["annotated_frames"] = sorted(annotated)


def flatten_masks_smallest_wins(
    masks: np.ndarray,
    label_ids: np.ndarray,
) -> np.ndarray:
    """Flatten (N, H, W) masks of one frame into an (H, W) label image.

    The pixels of mask ``i`` take the value ``label_ids[i]``, and 0 is
    background. Masks are painted from largest to smallest area, so
    where two masks overlap the smaller one keeps the shared pixels
    ("smallest_wins" occlusion policy).
    """
    masks = masks.astype(bool)
    label_frame = np.zeros(masks.shape[1:], dtype=np.uint16)
    areas = masks.reshape(len(masks), -1).sum(axis=1)
    for i in np.argsort(-areas, kind="stable"):
        label_frame[masks[i]] = label_ids[i]
    return label_frame


def compute_mask_props_dataset(
    labels_zarr: zarr.Array | np.ndarray,
    ds_bboxes: xr.Dataset,
) -> xr.Dataset:
    """Compute mask properties dataset.

    Masks are read from ``labels_zarr`` one frame at a time. Measuring the
    stored masks, rather than the label images in memory during prediction,
    is deliberate: the values can be reproduced from the store alone, and
    this function can run on an existing store.

    The mask with label ``k + 1`` is measured and stored at position
    ``k`` along ``id``, so each value sits at the (image_id, id) of its
    bounding box.

    The ellipse is the one with the same second moments as the mask, as
    defined by scikit-image. The returned dataset has these data variables,
    with NaN for slots with no mask:

    - ``mask_centroid`` (image_id, space, id): mean (x, y) of the mask
      pixels, which is also the ellipse centre.
    - ``mask_area``: number of mask pixels.
    - ``mask_area_to_convex_hull_area``: mask area / area of its convex
      hull (scikit-image's ``solidity``).
    - ``mask_area_to_mask_bbox_area``: mask area / area of the mask's own
      axis-aligned bounding box (scikit-image's ``extent``). Note this is
      the bounding box of the mask, not the annotated (prompt) bounding
      box, so it depends on the mask's rotation.
    - ``ellipse_major_axis_length``, ``ellipse_minor_axis_length``: full
      axis lengths in pixels, not semi-axes.
    - ``ellipse_major_axis_theta``: angle of the major axis from the image
      x-axis (columns) towards the y-axis (rows), in radians, in [0, pi).
      The image y-axis points down, so a positive angle looks clockwise
      on screen. It is arbitrary for near-circular masks.
    - ``ellipse_eccentricity``: 0 for a circle, towards 1 for an elongated
      ellipse.
    """
    # Get n images and max n annotations from ds_bboxes
    n_images, n_ids = ds_bboxes.sizes["image_id"], ds_bboxes.sizes["id"]

    # Initialise one (image_id, id) array per scalar variable, and an
    # (image_id, space, id) array for the centroid; NaN where no mask
    scalar_var_names = [*PROPS_TO_VARS.values(), "ellipse_major_axis_theta"]
    scalar_var_arrays = {
        name: np.full((n_images, n_ids), np.nan) for name in scalar_var_names
    }
    centroid_array = np.full((n_images, 2, n_ids), np.nan)  # 2 for space dim

    # Loop thru images
    for f in range(n_images):
        # Compute geom properties
        table = regionprops_table(
            np.asarray(labels_zarr[f]),
            properties=["label", "centroid", "orientation", *PROPS_TO_VARS],
        )

        # Fill in scalar variable arrays
        idx = table["label"] - LABEL_OFFSET
        for prop_name, var_name in PROPS_TO_VARS.items():
            scalar_var_arrays[var_name][f, idx] = table[prop_name]

        # Convert geometric properties to the x-axis
        # (scikit-image measures from the row axis)
        scalar_var_arrays["ellipse_major_axis_theta"][f, idx] = np.mod(
            np.pi / 2 - table["orientation"], np.pi
        )
        centroid_array[f, 0, idx] = table["centroid-1"]  # x = column
        centroid_array[f, 1, idx] = table["centroid-0"]  # y = row

    return xr.Dataset(
        {
            **{
                name: (("image_id", "id"), arr)
                for name, arr in scalar_var_arrays.items()
            },
            "mask_centroid": (
                ("image_id", "space", "id"),
                centroid_array,
            ),
        },
        coords=ds_bboxes[["image_id", "space", "id"]].coords,
    )


def append_mask_props(path_to_zarr: str | Path, ds_props: xr.Dataset) -> None:
    """Add the data variables of ``ds_props`` to an existing store.

    ``to_zarr`` writes values by position, without aligning them to the
    stored coordinates, so ``ds_props`` must follow the store's
    ``image_id`` and ``id`` order (it does when built by
    ``compute_mask_props`` from the same ``ds_bboxes``).
    """
    # drop coordinates from ds_props before appending, to not
    # overwrite coordinates of the zarr store
    ds_props = ds_props.drop_vars(list(ds_props.coords))

    # copy attrs from store
    # (fast, it only reads root metadata file)
    ds_props.attrs = dict(zarr.open_group(path_to_zarr).attrs)

    # append ds_props to zarr store
    # (to_zarr(mode="a") replaces the store's root attrs)
    ds_props.to_zarr(path_to_zarr, mode="a")


def main(args: argparse.Namespace) -> None:
    """Generate masks from bbox annotations using SAM2."""
    # Imports not needed for tests
    from ethology.io.annotations import load_bboxes
    from sam2.sam2_image_predictor import SAM2ImagePredictor

    # Parse CLI args
    data_dir = args.data_dir
    images_dir = data_dir / "frames"
    annotations_dir = data_dir / "annotations"
    annotations_file = annotations_dir / "VIA_JSON_combined_coco_gen.json"
    batch_size = args.batch_size
    model_name = args.model

    # Set device
    if torch.cuda.is_available():
        device = "cuda"
    elif torch.backends.mps.is_available():
        device = "mps"
    else:
        device = "cpu"
    print(f"Using device: {device}")

    # Define output zarr store, holding bboxes, masks and mask parameters
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_zarr = annotations_dir / f"bboxes_and_masks_{timestamp}.zarr"

    # Read groundtruth as an ethology annotation dataset
    ds_bboxes = load_bboxes.from_files(
        annotations_file,
        format="COCO",
        images_dirs=images_dir,
    )

    # Load ground truth images as a lazy array
    # (aligned with the image_id coordinate in ds_bboxes)
    img_paths = get_image_paths(ds_bboxes)
    image_array = ImageArrayLazy(img_paths)

    # Load SAM2 predictor on device
    image_predictor = SAM2ImagePredictor.from_pretrained(
        model_id=model_name,
        device=device,
    )

    # Create zarr store with the bboxes, and an empty array
    # for the output ID-encoded masks
    image_h, image_w = image_array.shape[1:3]
    metadata_dict = {
        "timestamp": timestamp,
        "sam2_model": model_name,
        "source_images_dir": str(images_dir),
        "annotations_file": str(annotations_file),
        "annotations_format": "COCO",
        "batch_size": batch_size,
        "multimask_output": False,
        "prompt_type": "bounding_box",
        "mask_encoding": "instance_id",
        "background_label": 0,
        "id_offset": LABEL_OFFSET,
        "occlusion_policy": "smallest_wins",
        "device": device,
    }
    create_dataset_store(
        output_zarr,
        ds_bboxes,
        (image_h, image_w),
        metadata_dict=metadata_dict,
    )

    # Get labels zarr array
    # (one frame per chunk)
    labels_zarr = zarr.open_array(output_zarr, path="labels")

    # Extract bbox prompts and label IDs per frame,
    # as dict mapping frame index to a tuple of
    # (N, 4) bbox arrays in [x1, y1, x2, y2] format
    # and (N,) arrays with the mask label ID associated to each box
    map_frame_idx_to_boxes_and_label_ids = get_bboxes_and_label_ids_per_frame(
        ds_bboxes
    )

    # Predict masks in image batches
    # and write to zarr store in OCTRON-format
    predict_masks_across_images(
        image_predictor,
        image_array,
        map_frame_idx_to_boxes_and_label_ids,
        labels_zarr,
        batch_size,
    )

    # Add geom parameters per bbox to zarr store
    ds_props = compute_mask_props_dataset(labels_zarr, ds_bboxes)
    append_mask_props(output_zarr, ds_props)

    print(f"Saved bboxes, masks and mask parameters to {output_zarr}")


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
