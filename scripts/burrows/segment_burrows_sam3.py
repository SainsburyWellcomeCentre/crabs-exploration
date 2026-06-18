"""Segment burrows with SAM3 from per-video point prompts.

For every PNG frame in the input directory, we look up the manual point prompts
for its video, run SAM3 image inference (optionally combined with a text
prompt), and postprocess the predicted masks. Postprocessing splits each mask
into connected regions and keeps only those within an area range and above a
solidity threshold.

The surviving regions are written, per frame, into a single ID-encoded mask
zarr store (background = 0, regions = 1, 2, ...; higher ID wins on overlap),
alongside a sibling array with the SAM3 confidence score per region.

The store is timestamped and laid out as:
    masks_<YYYYMMDD_HHMMSS>.zarr
    ├── masks   (n_frames, H, W) int16   # ID-encoded masks
    └── scores  (n_frames, max_regions + 1) float32  # score per region ID

The input prompt CSV is the one produced by annotate_burrow_prompts_manual.py,
with columns:
    group_id,        # "<video>_mean_n<frame>.png"; video = part before "_mean"
    prompt_point_x,
    prompt_point_y

Usage (dependencies are auto-installed via uv):
* Default (text prompt "hole")
    uv run segment_burrows_sam3.py /path/to/images_dir \
        /path/to/manual_prompts.csv /path/to/out_dir
* Disable the text prompt (rely on point prompts only)
    uv run segment_burrows_sam3.py /path/to/images_dir \
        /path/to/manual_prompts.csv /path/to/out_dir --text-prompt ""
* Custom postprocessing thresholds
    uv run segment_burrows_sam3.py /path/to/images_dir \
        /path/to/manual_prompts.csv /path/to/out_dir \
        --min-mask-area-pixels 100 --max-mask-area-pixels 3000

Follows the official SAM3 image predictor example:
https://github.com/facebookresearch/sam3/blob/main/examples/sam3_image_predictor_example.ipynb
and https://github.com/facebookresearch/sam3#basic-usage
"""

# /// script
# requires-python = ">=3.11"
# dependencies = [
#   "Pillow",
#   "zarr",
#   "torch>=2.5.1",
#   "torchvision>=0.20.1",
#   "pycocotools",
#   "sam3 @ git+https://github.com/facebookresearch/sam3.git",
#   "einops",
#   "huggingface_hub",
#   "scikit-image",
#   "numpy",
#   "pandas",
#   "psutil",  # undeclared transitive dep of sam3's video predictor
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
import contextlib
import os

# Reduce CUDA allocator fragmentation. Must be set before torch is imported.
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

import sys
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import zarr
from PIL import Image
from sam3 import build_sam3_image_model
from sam3.model.sam3_image_processor import Sam3Processor
from skimage.measure import label as sk_label
from skimage.measure import regionprops


class ImageArrayLazy:
    """A lazy array for images passed as a list."""

    def __init__(self, img_paths: list[Path]):
        """Store sorted image paths and cache the shared image shape."""
        # add sorted list of paths
        self.img_paths = sorted(img_paths)

        # add image shape, assuming all have same as first sample
        sample = np.array(Image.open(img_paths[0]))  # H, W, C
        self.img_h, self.img_w, self.img_c = sample.shape

    def __len__(self):
        """Return the number of images."""
        return len(self.img_paths)

    def __getitem__(self, idx: int):
        """Load and return the image at ``idx`` as an array."""
        return np.array(Image.open(self.img_paths[idx]))

    @property
    def shape(self):
        """Return the stack shape (B, H, W, C)."""
        return (len(self.img_paths), self.img_h, self.img_w, self.img_c)


def _initialise_mask_zarr(
    output_dir: Path,
    image_shape: tuple[int, int, int],
    metadata_dict: dict,
    max_regions_per_image: int,
) -> tuple[zarr.Group, Path]:
    """Create a timestamped ID-encoded mask zarr store with metadata.

    The store holds a ``masks`` array (one ID-encoded mask per frame) and a
    sibling ``scores`` array (one SAM3 confidence score per region ID). Mask
    instance IDs start at 1, since 0 is reserved for the background label, so
    the ``scores`` array has ``max_regions_per_image + 1`` columns (column 0,
    the background, is unused and left as NaN).
    """
    # Create a timestamped masks zarr store in the output directory
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir.mkdir(parents=True, exist_ok=True)
    output_masks_zarr = output_dir / f"masks_pass_1_{timestamp}.zarr"

    n_images, image_h, image_w = image_shape

    # Initialise root
    root = zarr.open_group(output_masks_zarr, mode="w")

    # Add ID-encoded mask array
    root.create_array(
        "masks",
        shape=(n_images, image_h, image_w),
        dtype="int16",
        fill_value=0,  # background
        chunks=(1, image_h, image_w),
    )

    # Initialise scores array as a sibling of "masks"
    # shape (n_frames, max_regions + 1); column 0 = background, unused -> NaN
    root.create_array(
        "scores",
        shape=(n_images, max_regions_per_image + 1),
        chunks=(1, max_regions_per_image + 1),
        dtype="float32",
        fill_value=np.nan,
    )

    # Add metadata to root
    root.attrs.update(metadata_dict)

    return root, output_masks_zarr


def _extract_normalised_point_prompts_per_video(
    manual_prompts_csv: Path | str, img_w: float, img_h: float
) -> dict:
    """Compute dict mapping video to normalised point prompt coordinates.

    The CSV ``group_id`` column has format ``<video>_mean_n<frame>.png``; the
    video string is everything before the first underscore. Coordinates are
    normalised by image width and height into ``[0, 1]``.
    """
    # Read from csv
    df_prompts = pd.read_csv(manual_prompts_csv)

    # point prompts in pixel xy ---> normalised, keyed by video string
    points_xy_normalised_per_video = {
        str(video).split("_")[0]: (
            group[["prompt_point_x", "prompt_point_y"]].to_numpy()
            / np.array([img_w, img_h])
        ).astype(np.float32)
        for video, group in df_prompts.groupby("group_id")
    }

    return points_xy_normalised_per_video


def _add_prompts_to_inference_state(
    processor, image, prompts_xy_norm, text_prompt=None
):
    """Add image, normalised points and optional text prompt to the state."""
    # Pass image to processor and reset inference state
    inference_state = processor.set_image(image)
    processor.reset_all_prompts(
        inference_state
    )  # mutates the state dict in place

    # Add optional text prompt
    if text_prompt is not None:
        inference_state = processor.set_text_prompt(
            state=inference_state, prompt=text_prompt
        )

    # Add normalised point prompts (all at once; grounds a single time)
    inference_state = _add_point_prompts(
        processor, inference_state, prompts_xy_norm, labels=True
    )

    return inference_state


def _add_point_prompts(processor, inference_state, points_xy, labels=True):
    """Add point prompts to the inference state.

    This function mirrors ``Sam3Processor.add_geometric_prompt`` but appends
    points as geometric prompts instead of a bounding box. It relies on SAM3
    internals (``_get_dummy_prompt`` and ``_forward_grounding``) as the
    processor does not expose a public point method.

    All points are appended and the grounding forward pass runs **once** at
    the end (rather than once per point). Re-running ``_forward_grounding``
    after every point recomputes full-resolution masks for all accumulated
    objects each time, which is wasteful and blows up GPU memory.

    Note: ``points_xy`` is an ``(N, 2)`` array of ``(x, y)`` pairs normalized
    to ``[0, 1]`` (a single ``(x, y)`` pair is also accepted). ``labels``
    sets each input point as positive (``True``, default) or negative
    (``False``); pass a scalar to apply the same label to all points, or an
    ``(N,)`` sequence for per-point labels.
    """
    # Throw error if image not passed thru backbone yet
    # (if passed, SAM3 stashes the backbone resulting features under the key
    # "backbone_out" in the state dict; we need an image to ground against).
    if "backbone_out" not in inference_state:
        raise ValueError("call processor.set_image before adding a prompt")

    # Check if a text prompt was passed
    if "language_features" not in inference_state["backbone_out"]:
        # if not: set text prompt to a dummy "visual" text so the
        # model relies only on the geometric prompt (this is the standard
        # way SAM3 signals this, see `add_geometric_prompt`)
        dummy_text_outputs = processor.model.backbone.forward_text(
            ["visual"], device=processor.device
        )
        inference_state["backbone_out"].update(dummy_text_outputs)

    # Ensure a geometric_prompt container exists to append the point into.
    # (we need to initialise it here because we append to it later)
    if "geometric_prompt" not in inference_state:
        inference_state["geometric_prompt"] = (
            processor.model._get_dummy_prompt()
            # returns an empty "Prompt" container with number of boxes = 0;
            # an empty Prompt container (zero boxes, batch=1) that subsequent
            # append_boxes/append_points calls grow into.
            #
            # Note: the SAM3 image processor also supports mask prompts
            # internally but this is not exposed in the public API. Masks as
            # prompts are used internally by its tracker/video stack, when the
            # previous frame mask is passed as prompt to the next frame.
        )

    # Add input point prompts with their labels to the inference state
    # NOTE: here batch=1, n_points = N (the number of points passed in)
    points_xy = np.atleast_2d(
        np.asarray(points_xy, dtype=np.float32)
    )  # (N, 2)
    n_points = points_xy.shape[0]

    point_coords = torch.tensor(
        points_xy,
        device=processor.device,
        dtype=torch.float32,
    ).view(n_points, 1, 2)  # (n_points, batch, 2)

    point_labels = torch.as_tensor(
        np.broadcast_to(labels, (n_points,)).copy(),
        device=processor.device,
        dtype=torch.bool,
    ).view(n_points, 1)  # (n_points, batch)

    inference_state["geometric_prompt"].append_points(
        point_coords,
        point_labels,
    )

    # Ground once over all accumulated prompts.
    return processor._forward_grounding(inference_state)


def _extract_sam3_results_one_img(inference_state):
    """Extract SAM3 boolean masks and scores for a single image.

    Masks are moved to CPU so this frame's GPU state can be released before
    the next frame: otherwise the previous state (backbone features +
    full-res masks_logits) stays alive during the next forward pass.
    """
    masks = inference_state["masks"].cpu().numpy()
    scores = inference_state["scores"].float().cpu().numpy()  # (N,)

    masks = masks.squeeze(1) if masks.ndim == 4 else masks  # (N, H, W)
    n_objects = masks.shape[0]

    return masks, scores, n_objects


def _postprocess_masks(
    masks: np.ndarray,
    scores: np.ndarray,
    min_area: int,
    max_area: int,
    min_solidity: float,
):
    """Split masks into connected regions and filter.

    Regions are filtered based on area range and solidity.

    ``masks`` is (N, H, W) boolean. Returns ``(id_encoded_mask, surviving_ids,
    surviving_scores, drop_counts)`` where ``id_encoded_mask`` is an
    ID-encoded mask, ``surviving_scores`` carries the source object's score
    onto each kept mask,
    and ``drop_counts`` is a dict counting how many connected regions were
    dropped per reason.

    The score mapping is needed because the function splits masks into
    connected components (regions), so a single SAM3 object can yield several
    kept masks (or none), and each must inherit the right score.
    """
    kept_regions_bool_masks = []
    list_mask_idcs = []
    drop_counts = {"area_low": 0, "area_high": 0, "solidity": 0}

    # Loop thru masks
    for mask_idx, mask in enumerate(masks.astype(bool)):
        # Label connected regions in mask
        # (connected regions are assigned the same int)
        label_mask = sk_label(mask)

        # Compute properties per region
        list_regions_w_props = regionprops(label_mask)

        # Loop thru regions
        for region in list_regions_w_props:
            # Filter by min area
            if region.area < min_area:
                drop_counts["area_low"] += 1
                continue

            # Filter by max area
            if region.area > max_area:
                drop_counts["area_high"] += 1
                continue

            # Filter by solidity (proxy for blob-likeness)
            # (ratio of pixels in the region to pixels of the convex hull,
            # ranges from 0 (theoretical) to 1 (perfectly convex))
            # convex_area >= area >= 1 for any real region
            if region.solidity < min_solidity:
                drop_counts["solidity"] += 1
                continue

            # If all pass: retain that region within the mask
            kept_regions_bool_masks.append(label_mask == region.label)
            # Keep track of the mask ID associated to this region too
            list_mask_idcs.append(mask_idx)

    # Get list of scores for kept regions
    kept_regions_scores = scores[list_mask_idcs]

    # -------------------------------
    # Compute id-encoded mask per region
    # boolean masks (N, H, W) -> ID-encoded (H, W); higher ID wins
    id_encoded_mask, list_surviving_ids = _convert_bool_to_id_mask(
        kept_regions_bool_masks, masks.shape[1:]
    )
    surviving_scores = kept_regions_scores[list_surviving_ids - 1]

    # Log
    drop_counts["id_overlap"] = len(list_mask_idcs) - len(list_surviving_ids)

    return id_encoded_mask, list_surviving_ids, surviving_scores, drop_counts


def _convert_bool_to_id_mask(list_kept_region_masks, img_h_w):
    """Express list of boolean mask arrays as a single ID-encoded mask.

    Higher ID wins on overlap. Returns the ID-encoded mask and the
    ``surviving_ids``: the non-background IDs that still have at least one
    pixel in the final mask.
    """
    # initialise id-encoded mask with all zeros
    id_encoded_mask = np.zeros(img_h_w, dtype=np.int16)

    # Paint each region in, assigning IDs from 1 upward (0 = background)
    # Note that regions with higher ID will win in an overlap
    for mask_idx, bool_mask in enumerate(list_kept_region_masks, start=1):
        id_encoded_mask[bool_mask] = mask_idx

    # Compute the final IDs that survived the "higher ID wins" overlap collapse
    surviving_ids = np.unique(id_encoded_mask)
    surviving_ids = surviving_ids[surviving_ids != 0]

    return id_encoded_mask, surviving_ids


def _relabel_id_encoded_mask_to_dense(
    id_encoded_mask,
):
    """Relabel old IDs -> dense 1..M.

    This is so they fit the scores width and have no gaps.
    """
    # get old mask ids
    old_mask_ids = np.asarray(
        [id for id in np.unique(id_encoded_mask) if id != 0]
    )
    n_old_mask_ids = len(old_mask_ids)

    # map old (array index) to new (array value) IDs
    old_to_new_ids = np.zeros(id_encoded_mask.max() + 1, dtype=np.int16)
    old_to_new_ids[old_mask_ids] = np.arange(
        1, n_old_mask_ids + 1, dtype=np.int16
    )

    # apply map to id_encoded_mask
    new_id_encoded_mask = old_to_new_ids[
        id_encoded_mask
    ]  # background (0) -> 0
    new_ids = np.arange(1, n_old_mask_ids + 1)

    return new_id_encoded_mask, new_ids


def main(args: argparse.Namespace) -> None:
    """Run SAM3 burrow segmentation per frame and write masks to zarr."""
    # ------------------------------------------------------------------
    # Load frames as a lazy array and map each frame to its video
    list_image_files = sorted(Path(args.images_dir).glob("*.png"))
    image_array = ImageArrayLazy(list_image_files)
    print(f"Loaded {len(image_array)} frames of shape {image_array.shape[1:]}")

    list_video_per_img = [
        img_p.stem.split("_", 1)[0].split("-Loop")[0]
        for img_p in list_image_files
    ]

    # ------------------------------------------------------------------
    # Initialise the output ID-encoded mask zarr store (includes scores)
    image_shape = image_array.shape[:3]
    n_images, image_h, image_w = image_shape
    metadata_dict = {
        "sam3_model": "sam3_image",
        "source_images_dir": str(args.images_dir),
        "manual_prompts_csv": str(args.manual_prompts_csv),
        "n_images": n_images,
        "image_shape": [image_h, image_w],
        "estim_max_regions_per_image": args.max_regions_per_image,
        "text_prompt": args.text_prompt,
        "sam3_confidence_threshold": args.conf_threshold,
        "postproc_min_mask_area_PIXELS": args.min_mask_area_pixels,
        "postproc_max_mask_area_PIXELS": args.max_mask_area_pixels,
        "postproc_min_solidity": args.min_solidity,
        "mask_encoding": "instance_id",
        "background_label": 0,
        "id_first_index": 1,
        # mask instance IDs in the zarr store start at 1 (not 0),
        # because 0 is reserved for the background label.
    }
    root, output_masks_zarr = _initialise_mask_zarr(
        Path(args.output_dir),
        image_shape,
        metadata_dict,
        args.max_regions_per_image,
    )

    # ------------------------------------------------------------------
    # Extract point prompts per video (normalised coords, keyed by video)
    points_xy_normalised_per_video = (
        _extract_normalised_point_prompts_per_video(
            args.manual_prompts_csv, image_array.img_w, image_array.img_h
        )
    )

    # ------------------------------------------------------------------
    # Build SAM3 image model and processor under inference/autocast contexts.
    # autocast avoids a mismatch between bfloat16 (model params) and float
    # (input); it is only enabled when running on CUDA.
    autocast_ctx = (
        torch.autocast("cuda", dtype=torch.bfloat16)
        if torch.cuda.is_available()
        else contextlib.nullcontext()
    )
    with torch.inference_mode(), autocast_ctx:
        # Instantiate model (holds weights and submodules: backbone, text
        # encoder, grounding head, mask decoder) and the processor (a wrapper
        # handling I/O conversion, inference_state mutation, prompt
        # accumulation and confidence thresholding; model is at
        # ``processor.model``).
        model = build_sam3_image_model()
        processor = Sam3Processor(
            model, confidence_threshold=args.conf_threshold
        )

        # --------------------------------------------------------------
        # Run inference on every frame and write ID-encoded masks to zarr
        count_postproc_frames_empty = 0
        postproc_frames_w_masks = []
        # (frame_idx, n_masks) for every frame run through inference, including
        # empty ones (0 masks); excludes frames skipped for having no prompts.
        n_masks_per_frame = []

        for frame_idx in range(len(image_array)):
            # Load image
            image = Image.fromarray(image_array[frame_idx])

            # Get point prompts normalised for the corresponding video
            video_str = list_video_per_img[frame_idx]
            prompts_xy_norm = points_xy_normalised_per_video.get(video_str)
            if prompts_xy_norm is None or len(prompts_xy_norm) == 0:
                print(f"Frame {frame_idx} ({video_str}): no prompts, skipping")
                continue

            # Add all prompts to inference state
            inference_state = _add_prompts_to_inference_state(
                processor,
                image,
                prompts_xy_norm,
                text_prompt=args.text_prompt,
            )

            # Get predicted boolean masks and scores, then release GPU state
            masks, scores, n_objects = _extract_sam3_results_one_img(
                inference_state
            )
            del inference_state
            torch.cuda.empty_cache()

            # Log if no detections
            if n_objects == 0:
                print(f"Frame {frame_idx} ({video_str}): no detections")
                n_masks_per_frame.append((frame_idx, 0))
                continue

            # Postprocess SAM3-predicted masks:
            # - Split masks into "regions",
            # - Filter out masks whose area is out of bounds,
            # - Flatten overlaps via ID-encoding
            (
                id_encoded_mask,
                surviving_ids,
                surviving_scores,
                drop_counts,
            ) = _postprocess_masks(
                masks,
                scores,
                args.min_mask_area_pixels,
                args.max_mask_area_pixels,
                args.min_solidity,
            )

            # -------------------------
            # Log postprocessing results for this frame
            n_surviving_regions = len(surviving_ids)
            if n_surviving_regions == 0:
                print(
                    f"Frame {frame_idx} ({video_str}): "
                    "no masks after postprocessing"
                )
                count_postproc_frames_empty += 1
                n_masks_per_frame.append((frame_idx, 0))
                continue

            n_total_regions = n_surviving_regions + sum(drop_counts.values())
            print(
                f"Frame {frame_idx} ({video_str}): postprocessing kept "
                f"{n_surviving_regions}/{n_total_regions} regions, "
                f"dropped {drop_counts} (all before capping)."
            )
            # -------------------------

            # Enforce the per-frame cap on max number of regions,
            if n_surviving_regions > args.max_regions_per_image:
                # we select the top M scoring ones
                capped_sorted_idcs = np.argsort(surviving_scores)[::-1][
                    : args.max_regions_per_image
                ]
                # Get corresponding "M" IDs and scores in score-order!
                selected_ids = surviving_ids[capped_sorted_idcs]
                selected_scores = surviving_scores[capped_sorted_idcs]

                # Drop non-selected IDs from the mask
                id_encoded_mask[~np.isin(id_encoded_mask, selected_ids)] = 0

                # Re-sort the scores into ascending ID order from the selected
                # IDs; this is required for the relabel step
                order = np.argsort(selected_ids)
                surviving_scores = selected_scores[order]

            # ----------------
            # Relabel old IDs -> dense 1..M so zarr array has no gaps
            new_id_encoded_mask, new_ids = _relabel_id_encoded_mask_to_dense(
                id_encoded_mask
            )

            # Save results to zarr
            root["masks"][frame_idx] = new_id_encoded_mask
            root["scores"][frame_idx, new_ids] = surviving_scores

            # -------------------------
            # Log final number of masks
            n_masks_saved = len(new_ids)
            n_masks_per_frame.append((frame_idx, n_masks_saved))
            print(f"Frame {frame_idx} ({video_str}): {n_masks_saved} masks")

            postproc_frames_w_masks.append(frame_idx)

    # Add extra metrics to zarr array
    root.attrs["frames_with_masks"] = postproc_frames_w_masks
    root.attrs["n_masks_per_frame"] = n_masks_per_frame
    print(f"Saved ID-encoded mask zarr to {output_masks_zarr}")

    # ----------------------------
    # Summarise masks-per-frame statistics over every frame run through
    # inference, including empty frames (0 masks).
    print(
        f"N frames with no masks after postprocessing: "
        f"{count_postproc_frames_empty}"
    )
    if n_masks_per_frame:
        counts = np.array([n_masks for _, n_masks in n_masks_per_frame])
        mean_masks = counts.mean()
        print(
            f"Stats for masks per frame (n={len(counts)} frames)"
            f"mean={mean_masks:.2f}, "
            f"median={np.median(counts):.1f}, "
            f"min={counts.min()}, max={counts.max()}"
        )

        # Frame indices with fewer than the mean number of masks per frame
        frames_below_mean = [
            frame_idx
            for frame_idx, n_masks in n_masks_per_frame
            if n_masks < mean_masks
        ]
        print(
            f"Frames with fewer than mean ({mean_masks:.2f}) masks per frame "
            f"({len(frames_below_mean)} frames): {frames_below_mean}"
        )


def parse_args(list_args: list[str]) -> argparse.Namespace:
    """Parse CLI args."""
    parser = argparse.ArgumentParser(
        description=(
            "Segment burrows with SAM3 from per-video point prompts and "
            "write the postprocessed, ID-encoded masks to a zarr store."
        ),
    )
    parser.add_argument(
        "images_dir",
        type=Path,
        help="Directory of PNG frames to run inference on.",
    )
    parser.add_argument(
        "manual_prompts_csv",
        type=Path,
        help=(
            "CSV of per-video manual point prompts, as produced by "
            "annotate_burrow_prompts_manual.py (columns: group_id, "
            "prompt_point_x, prompt_point_y)."
        ),
    )
    parser.add_argument(
        "output_dir",
        type=Path,
        help=(
            "Output directory. A timestamped "
            "'masks_pass_1_<YYYYMMDD_HHMMSS>.zarr' "
            "store is created inside it, so multiple runs never collide."
        ),
    )
    parser.add_argument(
        "--text-prompt",
        type=str,
        default="hole",
        help=(
            "Text prompt passed to SAM3 alongside the point prompts "
            "(default: 'hole'). Pass an empty string to disable the text "
            "prompt and rely on the point prompts only."
        ),
    )
    parser.add_argument(
        "--conf-threshold",
        type=float,
        default=0.35,
        help=(
            "SAM3 confidence threshold; masks below this score are not scaled "
            "up to full resolution (default: 0.35)."
        ),
    )
    parser.add_argument(
        "--min-mask-area-pixels",
        type=int,
        default=200,
        help=(
            "Connected regions smaller than this area (in pixels) are dropped "
            "during postprocessing (default: 200)."
        ),
    )
    parser.add_argument(
        "--max-mask-area-pixels",
        type=int,
        default=2500,
        help=(
            "Connected regions larger than this area (in pixels) are dropped "
            "during postprocessing (default: 2500)."
        ),
    )
    parser.add_argument(
        "--min-solidity",
        type=float,
        default=0.95,
        help=(
            "Minimum region solidity (region area / convex hull area, in "
            "[0, 1]) kept during postprocessing (default: 0.95)."
        ),
    )
    parser.add_argument(
        "--max-regions-per-image",
        type=int,
        default=500,
        help=(
            "Estimated upper bound on kept regions per frame; sets the number "
            "of columns of the scores array (default: 500)."
        ),
    )

    args = parser.parse_args(list_args)

    # An empty text prompt disables the text prompt entirely.
    if args.text_prompt == "":
        args.text_prompt = None

    return args


if __name__ == "__main__":
    args = parse_args(sys.argv[1:])
    main(args)
