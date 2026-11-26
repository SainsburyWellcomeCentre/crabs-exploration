"""Mask tracked crabs in a video with SAM2."""

import argparse
import logging
import sys
from datetime import datetime
from pathlib import Path

import cv2
import dask.array as da
import numpy as np
import torch
import xarray as xr
import zarr
from zarr.codecs import TransposeCodec, ZstdCodec

from crabs.mask.mask_props import compute_mask_props_of_video
from crabs.mask.utils.bbox_prompts import (
    copy_video_bboxes_to_masks,
    extract_bboxes_from_clip,
)
from crabs.utils.io import (
    get_video_parameters,
    open_video,
    parse_video_frame_reading_error_and_log,
    read_yaml,
)

DEFAULT_MASK_CONFIG = str(
    Path(__file__).parent / "config" / "mask_config.yaml"
)

# 0 is background in a label image, so crab labels start at 1
BACKGROUND_LABEL = 0

# Compressor of the label images.
# (writes are slower for higher levels, so we use the default for the rest
# of the arrays that are not label images)
LABELS_COMPRESSOR = ZstdCodec(level=15)

# Store each `labels` chunk with time last, so each pixel's values over
# the chunk's frames are contiguous and compress better. zarr undoes it
# on read.
LABELS_TIME_LAST = TransposeCodec(order=(0, 2, 3, 1))

# Pytorch accelerator names that differ from their torch device name
ACCELERATOR_TO_DEVICE = {"gpu": "cuda"}


def load_full_mask_config(config_file: str) -> dict:
    """Load and check the masking config, with defaults for missing keys.

    The default config file (`DEFAULT_MASK_CONFIG`) is the single source
    of default values and of valid keys.

    Raises
    ------
    ValueError
        If the file sets a key that is not a masking parameter (e.g. a
        misspelled or renamed one, which would otherwise be ignored), if
        `n_frames_per_shard` is not a multiple of `n_frames_per_chunk`,
        or if `occlusion_policy` is not one that is implemented.

    """
    default_config = read_yaml(DEFAULT_MASK_CONFIG)
    input_config = read_yaml(config_file)
    unrecognised_keys = sorted(set(input_config) - set(default_config))
    if unrecognised_keys:
        raise ValueError(
            f"Unknown keys {unrecognised_keys} in {config_file}. "
            f"Valid keys are {sorted(default_config)}."
        )
    config = {**default_config, **input_config}

    # Check n_frames_per_shard is a multiple of n_frames_per_chunk
    # (zarr would also raise an error, but we want to fail earlier)
    n_frames_per_chunk = config["n_frames_per_chunk"]
    n_frames_per_shard = config["n_frames_per_shard"]
    if n_frames_per_shard and n_frames_per_shard % n_frames_per_chunk:
        raise ValueError(
            f"n_frames_per_shard ({n_frames_per_shard}) must be a multiple "
            f"of n_frames_per_chunk ({n_frames_per_chunk})."
        )

    # Check occlusion policy
    if config["occlusion_policy"] != "smallest_wins":
        raise ValueError(
            f"Unknown occlusion_policy '{config['occlusion_policy']}'. The "
            "only policy implemented is 'smallest_wins'."
        )
    return config


def select_video_nodes(
    trajectories_store: Path, match: str
) -> list[xr.DataTree]:
    """Return the matching video groups from the trajectories store.

    Raises
    ------
    ValueError
        If `trajectories_store` is not a zarr store, or if `match` selects no
        video group.

    """
    if trajectories_store.suffix != ".zarr":
        raise ValueError(
            f"--boxes must be a trajectories zarr store (.zarr), but "
            f"'{trajectories_store.name}' has suffix "
            f"'{trajectories_store.suffix}'."
        )

    # Read trajectories data as a datatree
    dt_trajs = xr.open_datatree(trajectories_store, engine="zarr", chunks={})

    # Select relevant subset of videos.
    # (a pattern that matches nothing leaves only the root node behind)
    video_nodes = [
        node for node in dt_trajs.match(match).leaves if node.path != "/"
    ]
    if not video_nodes:
        raise ValueError(
            f"--match '{match}' selected no video group of "
            f"{trajectories_store}."
        )
    return video_nodes


def check_clips_paths(
    videos_dir: str, video_id: str, clip_ids: list[str]
) -> list[Path]:
    """Compute and check existence of the path to each clip in a video.

    Raises
    ------
    FileNotFoundError
        If any clip video is missing from `videos_dir`.

    """
    # List of paths to clips
    clip_videos = [
        Path(videos_dir) / f"{video_id}-{clip_id}.mp4" for clip_id in clip_ids
    ]
    # Check existence
    for clip_video in clip_videos:
        if not clip_video.exists():
            raise FileNotFoundError(f"Clip video not found: {clip_video}")
    return clip_videos


def load_sam2_predictor(model_id: str, device: str):
    """Import sam2 lazily and return a SAM2 image predictor."""
    try:
        from sam2.sam2_image_predictor import SAM2ImagePredictor
    except ImportError as e:
        raise ImportError(
            "mask-tracked-crabs needs SAM2. "
            "SAM2's build imports torch, so install crabs first and then "
            "the `masks` extra. To do so, from a checkout of the repo, run:\n"
            "  uv pip install .\n"
            "  uv pip install setuptools\n"
            '  uv pip install --no-build-isolation-package sam-2 ".[masks]"'
        ) from e
    return SAM2ImagePredictor.from_pretrained(model_id, device=device)


def get_store_path(
    args: argparse.Namespace, boxes_path: Path, timestamp: str
) -> Path:
    """Return the path to the output store, creating its parent directory.

    It is `--output_store` if given, or else a timestamped store under
    `--output_dir`.
    """
    if args.output_store:
        store_path = Path(args.output_store)
    else:
        store_path = (
            Path(args.output_dir) / f"{boxes_path.stem}_masks_{timestamp}.zarr"
        )
    store_path.parent.mkdir(parents=True, exist_ok=True)
    return store_path


def get_store_metadata(
    timestamp: str, device: str, boxes_path: Path, mask_config: dict
) -> dict:
    """Return the attributes shared by every video group this run writes.

    Each group also gets its own `source_video`, added in
    `setup_video_mask_group`.
    """
    return {
        "timestamp": timestamp,
        "sam2_model": mask_config["sam2_model_id"],
        "device": device,
        "boxes": {
            "file": str(boxes_path),
            "source": "trajectories_zarr",
            "ids": "trajectories_store_individual",
        },
        "background_label": BACKGROUND_LABEL,
        "occlusion_policy": mask_config["occlusion_policy"],
        "prompt_type": "bounding_box",
        # config values of the run, with defaults for keys the
        # config file leaves out
        "mask_config": mask_config,
    }


def setup_video_mask_group(
    video_id: str,
    ds_video: xr.Dataset,
    clip_paths: list[Path],
    store_path: Path,
    store_metadata: dict,
    mask_config: dict,
    zarr_mode_group: str,
) -> tuple[zarr.Array, xr.DataArray]:
    """Check a video's clips and create its group in the mask store.

    The group is created with its metadata, coordinates and `label_of`,
    and with the prompt boxes copied from the trajectories store. No
    `labels` data is written yet.

    Parameters
    ----------
    video_id : str
        Name of the video group, mirroring the trajectories store.
    ds_video : xr.Dataset
        The video's dataset from the trajectories store.
    clip_paths : list[Path]
        Path to each clip video, in the order of `ds_video.clip_id`.
    store_path : Path
        Path to the mask zarr store.
    store_metadata : dict
        Attributes shared by every video group this run writes. The group's
        `source_video` attribute is added to them.
    mask_config : dict
        The full masking config, for the chunk and shard sizes.
    zarr_mode_group : str
        Mode to write the zarr group with.

    Returns
    -------
    tuple[zarr.Array, xr.DataArray]
        The group's `labels` array and its `label_of` mapping.

    Raises
    ------
    ValueError
        If the clips differ in frame size, or a clip video is shorter
        than its metadata length.

    """
    # Check clips and get the group's frame size and length
    image_shape, n_frames = _get_frame_shape_and_max_frames_in_video(
        clip_paths
    )
    _check_clip_lengths(ds_video, clip_paths)

    # Initialise mask store and labels array
    labels_array = _create_mask_store(
        store_path=store_path,
        video_id=video_id,
        clip_ids=[str(c) for c in ds_video.clip_id.values],
        n_frames=n_frames,
        # copied from the input trajectories store
        individuals=[str(i) for i in ds_video.individual.values],
        image_shape=image_shape,
        metadata_dict={
            **store_metadata,
            "source_video": [str(p) for p in clip_paths],
        },
        n_frames_per_chunk=mask_config["n_frames_per_chunk"],
        n_frames_per_shard=mask_config["n_frames_per_shard"],
        zarr_mode_group=zarr_mode_group,
    )

    # Copy the prompt boxes into the masks group
    copy_video_bboxes_to_masks(ds_video, store_path)

    # Read mapping from individual to mask labels
    # (skipping consolidated metadata, which may be stale during a run)
    label_of = xr.open_zarr(
        store_path,
        group=video_id,
        consolidated=False,
    ).label_of

    return labels_array, label_of


def _get_frame_shape_and_max_frames_in_video(
    clip_videos: list[Path],
) -> tuple[tuple[int, int], int]:
    """Read the common frame size and the max length of all clips in a video.

    Throws an error if clips have inconsistent frame sizes. The data is read
    from each clip's metadata, without decoding frames.
    """
    # Get clip params
    clip_params = {p.name: get_video_parameters(str(p)) for p in clip_videos}
    max_n_frames = max(c["total_frames"] for c in clip_params.values())

    # Check all clips share a single frame shape
    frame_shapes = {
        name: (c["frame_height"], c["frame_width"])
        for name, c in clip_params.items()
    }
    if len(set(frame_shapes.values())) > 1:
        raise ValueError(
            f"Clips of one video differ in frame size {frame_shapes}, "
            "but one mask store group holds a single frame size."
        )

    # Return common frame shape and max length of a clip
    return next(iter(frame_shapes.values())), max_n_frames


def _check_clip_lengths(ds_video: xr.Dataset, clip_paths: list[Path]) -> None:
    """Raise if a clip video has fewer frames than its metadata length.

    Otherwise SAM2 would stop at the video's last frame, and the later
    boxes would get no mask. Only the video headers are read.

    Raises
    ------
    ValueError
        If any clip video is shorter than its metadata length, computed
        from `clip_first_frame_0idx` and `clip_last_frame_0idx`.

    """
    n_frames_metadata = (
        ds_video.clip_last_frame_0idx.values
        - ds_video.clip_first_frame_0idx.values
        + 1
    )
    for clip_path, n_frames_clip in zip(
        clip_paths, n_frames_metadata, strict=True
    ):
        n_frames_video = get_video_parameters(str(clip_path))["total_frames"]
        if n_frames_video < n_frames_clip:
            raise ValueError(
                f"Clip video {clip_path} has {n_frames_video} frames, but "
                f"its metadata length is {int(n_frames_clip)} frames."
            )


def _create_mask_store(
    store_path: str | Path,
    video_id: str,
    clip_ids: list[str],
    n_frames: int,
    individuals: list[str],
    image_shape: tuple[int, int],
    metadata_dict: dict,
    n_frames_per_chunk: int,
    n_frames_per_shard: int | None,
    zarr_mode_group: str,
) -> zarr.Array:
    """Initialise one video group of the store and return its labels array.

    The group holds one label image per frame (a single integer array
    in which each crab's pixels carry that crab's own pixel value), and
    `label_of`, the individual to pixel value mapping.

    The group is written with `compute=False`, so metadata, coordinates
    and `label_of` are written to disk, but no `labels` data yet. Each
    clip then fills its own region of the returned zarr array.

    Parameters
    ----------
    store_path : str | Path
        Path to the mask zarr store.
    video_id : str
        Name of the zarr group to write, mirroring the trajectories store.
    clip_ids : list[str]
        The video's clips, in the order their masks will be written.
    n_frames : int
        Length of the time axis, i.e. the longest clip of the video.
    individuals : list[str]
        The video group's `individual` coordinate, copied from the
        trajectories store.
    image_shape : tuple[int, int]
        Frame height and width, in pixels.
    metadata_dict : dict
        Attributes to write on the group.
    n_frames_per_chunk : int
        Number of frames per chunk.
    n_frames_per_shard : int | None
        Number of frames per shard file, a multiple of
        `n_frames_per_chunk`. None disables sharding.
    zarr_mode_group : str
        Mode to write the zarr group with.

    Returns
    -------
    zarr.Array
        The `labels` array of the group just written.

    """
    height, width = image_shape
    chunks = (1, n_frames_per_chunk, height, width)
    shards = (
        (1, n_frames_per_shard, height, width) if n_frames_per_shard else None
    )

    # Initialise dataset for video
    ds = xr.Dataset(
        {
            "labels": (
                ("clip_id", "time", "img_h", "img_w"),
                # we use dask arrays to make `labels` lazy,
                # so `to_zarr(compute=False)` writes no data yet.
                # Its chunks must match zarr's write unit (the
                # shard, or the chunk without sharding), otherwise xarray
                # refuses the layout
                da.zeros(
                    (len(clip_ids), n_frames, height, width),
                    chunks=shards or chunks,
                    dtype=np.uint16,
                ),
            ),
            "label_of": (
                ("individual",),
                np.arange(1, len(individuals) + 1, dtype=np.uint16),
            ),
        },
        coords={
            "clip_id": np.array(clip_ids, dtype=str),
            "time": np.arange(n_frames),
            "individual": np.array(individuals, dtype=str),
        },
        attrs=metadata_dict,
    )

    # Set encoding params for labels
    encoding: dict = {
        "labels": {
            "chunks": chunks,
            "filters": [LABELS_TIME_LAST],
            "compressors": [LABELS_COMPRESSOR],
        }
    }
    if shards:
        encoding["labels"]["shards"] = shards

    # Write dataset to zarr groups
    # (the store is consolidated once, at the end of `main`)
    ds.to_zarr(
        store_path,
        group=video_id,
        mode=zarr_mode_group,
        encoding=encoding,
        compute=False,
        consolidated=False,
    )

    return zarr.open_group(store_path, use_consolidated=False)[
        f"{video_id}/labels"
    ]


def predict_and_write_video_labels(
    ds_video: xr.Dataset,
    clip_paths: list[Path],
    labels_array: zarr.Array,
    label_of: xr.DataArray,
    predictor,
    max_prompts_per_batch: int,
) -> None:
    """Predict label frames for every clip of a video and write them to store.

    Clips with no tracked boxes are skipped, so their region of
    `labels_array` stays background.

    Parameters
    ----------
    ds_video : xr.Dataset
        The video's dataset from the trajectories store, with the boxes
        used to prompt SAM2.
    clip_paths : list[Path]
        Path to each clip video, in the order of `ds_video.clip_id`.
    labels_array : zarr.Array
        The `labels` array of the video group, from `create_mask_store`.
    label_of : xr.DataArray
        The individual to pixel value mapping written by
        `create_mask_store`.
    predictor : SAM2ImagePredictor
        The predictor, loaded once by the caller and reused across clips.
    max_prompts_per_batch : int
        Number of box prompts passed to SAM2 at once.

    """
    # Loop thru clips in video
    # (the clip's position along `clip_id` is its row in `labels_array`)
    for clip_index, (clip_id, clip_path) in enumerate(
        zip(ds_video.clip_id.values, clip_paths, strict=True)
    ):
        logging.info(f"Masking {clip_path.stem}")

        # Get prompt boxes for this clip
        tracked_bboxes_dict = extract_bboxes_from_clip(
            ds_video.sel(clip_id=clip_id)
        )
        if not tracked_bboxes_dict:
            logging.info(f"{clip_path.stem}: no tracked boxes, skipping")
            continue

        # Predict masks and write to store
        _predict_and_write_clip_labels(
            video_clip_path=str(clip_path),
            tracked_bboxes_dict=tracked_bboxes_dict,
            labels_array=labels_array,
            clip_index=clip_index,
            label_of=label_of,
            predictor=predictor,
            max_prompts_per_batch=max_prompts_per_batch,
        )


def _predict_and_write_clip_labels(
    video_clip_path: str,
    tracked_bboxes_dict: dict,
    labels_array: zarr.Array,
    clip_index: int,
    label_of: xr.DataArray,
    predictor,
    max_prompts_per_batch: int,
) -> int:
    """Predict label frames for all frames in a clip and write them to store.

    Frames are read from the clip video one at a time. For each frame with
    tracked boxes, SAM2 predicts one mask per box. Bool masks are then
    combined into a single label image. Frames without boxes have the
    background label everywhere.

    Parameters
    ----------
    video_clip_path : str
        Path to the clip video to read pixels from.
    tracked_bboxes_dict : dict
        Map from clip frame index to
        ``{"tracked_boxes": (n, 4) float, "ids": (n,) labels}``. It may
        hold no key for a frame with no boxes, and any extra keys are
        ignored.
    labels_array : zarr.Array
        The `labels` array of the video group, from `create_mask_store`.
    clip_index : int
        Position of this clip along the `clip_id` axis.
    label_of : xr.DataArray
        The individual to pixel value mapping written by
        `create_mask_store`.
    predictor : SAM2ImagePredictor
        The predictor, loaded once by the caller and reused across clips.
    max_prompts_per_batch : int
        Number of box prompts passed to SAM2 at once.

    Returns
    -------
    int
        The number of frames read from the clip video.

    Raises
    ------
    RuntimeError
        If a frame cannot be read before the end of the clip video.

    """
    # Map individual name to the pixel value painted for its mask
    map_individual_to_label = {
        str(ind): int(lbl)
        for ind, lbl in zip(
            label_of.individual.values, label_of.values, strict=True
        )
    }

    # Initialise buffer: one shard of "label frames".
    # Frames are written to the store a whole shard at a time, so zarr
    # never has to read, modify and rewrite a shard file. It also keeps
    # memory bounded: only one shard of label frames is held at a time,
    # not the full clip's. With sharding disabled, the buffer is one
    # chunk.
    height, width = labels_array.shape[2:]
    buffer_n_frames = (labels_array.shards or labels_array.chunks)[1]
    buffer = np.zeros((buffer_n_frames, height, width), dtype=np.uint16)

    # Get this clip's length
    # (Note: the store's time axis is the group's longest clip, not a
    # specific clip length)
    clip_n_frames = get_video_parameters(video_clip_path)["total_frames"]

    # Loop thru frames in clip
    frame_idx = 0
    input_video_object = open_video(video_clip_path)
    while input_video_object.isOpened():
        # Read one frame
        ret, frame = input_video_object.read()
        if not ret:
            parse_video_frame_reading_error_and_log(frame_idx, clip_n_frames)
            # A read can fail before the last frame, e.g. if a slow
            # filesystem makes OpenCV time out. The frames not read would
            # be left unwritten, and read back as background, so we stop
            # rather than leave a partly masked clip that looks complete.
            if frame_idx < clip_n_frames:
                input_video_object.release()
                raise RuntimeError(
                    f"Failed to read frame {frame_idx} of "
                    f"{clip_n_frames} from {video_clip_path}."
                )
            break

        # Initialise output mask in buffer;
        # we initialise the slot for this frame because
        # the buffer is reused
        idx_in_buffer = frame_idx % buffer_n_frames
        buffer[idx_in_buffer, :, :] = BACKGROUND_LABEL

        # Predict masks using SAM2 and boxes prompt
        # (we use .get because there may be frames with no predicted crabs)
        frame_data = tracked_bboxes_dict.get(frame_idx)
        # skip frames with no boxes too, so SAM2's image encoder isn't run
        if frame_data is not None and len(frame_data["tracked_boxes"]) > 0:
            # we use index here: an id missing from `label_of` should fail
            # loudly rather than paint a wrong label
            box_labels = np.array(
                [map_individual_to_label[str(i)] for i in frame_data["ids"]]
            )
            _predict_and_flatten_masks_into(
                predictor,
                frame,
                frame_data["tracked_boxes"],
                box_labels,
                buffer[idx_in_buffer, :, :],
                max_prompts_per_batch,
            )

        # When idx_in_buffer == buffer_n_frames - 1, the buffer is full,
        # and so we write it into the final `labels_array` array
        if idx_in_buffer == buffer_n_frames - 1:
            labels_array[
                clip_index, frame_idx - idx_in_buffer : frame_idx + 1
            ] = buffer

        # Update frame index
        frame_idx += 1

    # After the loop is completed, frame_idx represents the number of
    # frames read.
    # If frame_idx % buffer_n_frames != 0, the last buffer/shard is
    # partially filled and has not been written to labels_array yet
    n_frames_in_buffer = frame_idx % buffer_n_frames
    if n_frames_in_buffer:
        start = frame_idx - n_frames_in_buffer
        labels_array[clip_index, start:frame_idx] = buffer[:n_frames_in_buffer]

    # Release video capture
    input_video_object.release()

    # Return number of frames actually read
    return frame_idx


def _predict_and_flatten_masks_into(
    predictor,
    frame_bgr: np.ndarray,
    boxes: np.ndarray,
    box_labels: np.ndarray,
    label_frame: np.ndarray,
    max_prompts_per_batch: int,
) -> None:
    """Compute label image from SAM2 prediction on one frame.

    `box_labels` are the pixel values to paint for these boxes, read from
    `label_of`, in the same order as `boxes`.

    The masks are flattened on SAM2's device (usually the GPU), so only
    the finished label image is copied back to CPU memory. Flattening
    each batch of prompts as it is predicted releases its full-frame
    masks before the next batch.

    The box prompts are sorted by area before batching, because a batch is
    painted before the next is predicted: without the sort, a large crab in
    a later batch would overwrite a small crab from an earlier one, which
    is the opposite of the "smallest wins" occlusion policy.
    """
    # SAM2 logs three INFO lines to the root logger on every set_image call,
    # so we silence INFO for that call only
    root_logger = logging.getLogger()
    previous_level = root_logger.level
    root_logger.setLevel(logging.WARNING)
    try:
        predictor.set_image(cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB))
    finally:
        root_logger.setLevel(previous_level)

    # Sort prompt boxes by size.
    # largest box first, so batch order agrees with the occlusion policy
    areas = (boxes[:, 2] - boxes[:, 0]) * (boxes[:, 3] - boxes[:, 1])
    order = np.argsort(-areas)
    boxes, box_labels = boxes[order], box_labels[order]

    # Initialise output tensor
    # int32, not uint16: torch implements few ops for unsigned types
    label_frame_on_device = torch.zeros(
        label_frame.shape,
        dtype=torch.int32,
        device=predictor.device,
    )

    # Loop thru batches of prompts
    for start in range(0, len(boxes), max_prompts_per_batch):
        box_batch = boxes[start : start + max_prompts_per_batch]
        label_batch = box_labels[start : start + max_prompts_per_batch]

        # Predict masks for this batch of boxes, (B, H, W) bool
        masks = _predict_masks_on_device(predictor, box_batch)

        # Flatten masks (occlusion policy)
        # smallest_wins: paint largest first, so where two masks overlap
        # the smaller crab keeps the contested pixels
        # (B, H, W) -> (B, H*W) -> sum -> (B,) area in pixels
        mask_areas = masks.flatten(start_dim=1).sum(dim=1)
        for i in torch.argsort(mask_areas, descending=True).tolist():
            label_frame_on_device.masked_fill_(masks[i], int(label_batch[i]))

    # Copy result into the caller's label image, in place
    # (int32 is cast to label_frame's uint16 on assignment)
    label_frame[:] = label_frame_on_device.cpu().numpy()


def _predict_masks_on_device(predictor, boxes: np.ndarray) -> torch.Tensor:
    """Predict SAM2 masks for the given box prompts, keeping them on device.

    SAM2's public `SAM2ImagePredictor.predict` copies every full-frame
    mask from the GPU to the CPU as float32. At 4K resolution, this is costly.
    To avoid this, this function calls the two private methods that
    `predict` wraps (`_prep_prompts` and `_predict`), but after them it leaves
    the masks on SAM2's device (usually the GPU).

    Returns
    -------
    torch.Tensor
        A (N, H, W) boolean tensor on SAM2's device, with one mask per box.

    """
    # move boxes to device and rescale to model input size
    _, _, _, unnorm_box = predictor._prep_prompts(
        None, None, boxes, None, normalize_coords=True
    )
    # predict on device
    masks, _, _ = predictor._predict(
        None, None, unnorm_box, None, multimask_output=False
    )
    return masks[:, 0]


def main(args: argparse.Namespace) -> None:
    """Predict masks with SAM2 from tracked boxes."""
    # Load and check inputs
    boxes_path = Path(args.boxes)
    mask_config = load_full_mask_config(args.mask_config_file)
    video_nodes = select_video_nodes(boxes_path, args.match)

    # Check every clip video exists before loading SAM2,
    # so a wrong --videos_dir fails in seconds, not after a model download
    map_video_to_clip_paths = {
        node.name: check_clips_paths(
            args.videos_dir,
            node.name,
            [str(c) for c in node.clip_id.values],
        )
        for node in video_nodes
    }

    # Initialise SAM2 predictor
    device = ACCELERATOR_TO_DEVICE.get(args.accelerator, args.accelerator)
    predictor = load_sam2_predictor(mask_config["sam2_model_id"], device)

    # Initialise output store
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    store_path = get_store_path(args, boxes_path, timestamp)
    zarr.open_group(store_path, mode=args.zarr_mode_store)
    store_metadata = get_store_metadata(
        timestamp, device, boxes_path, mask_config
    )

    # Loop thru video nodes
    for node in video_nodes:
        # Read video dataset
        video_id = node.name
        ds_video = node.to_dataset()
        clip_paths = map_video_to_clip_paths[video_id]

        # Create the video's group in the output store
        labels_array, label_of = setup_video_mask_group(
            video_id=video_id,
            ds_video=ds_video,
            clip_paths=clip_paths,
            store_path=store_path,
            store_metadata=store_metadata,
            mask_config=mask_config,
            zarr_mode_group=args.zarr_mode_group,
        )

        # Predict masks for every clip of the video and write to store
        predict_and_write_video_labels(
            ds_video=ds_video,
            clip_paths=clip_paths,
            labels_array=labels_array,
            label_of=label_of,
            predictor=predictor,
            max_prompts_per_batch=mask_config["max_prompts_per_batch"],
        )

        # Measure the masks and write the mask and ellipse parameters
        compute_mask_props_of_video(video_id, store_path)

    # Consolidate the store's metadata once, after every write of this run
    # (skipped if part of an array job, to avoid a race)
    if not args.no_consolidate:
        zarr.consolidate_metadata(store_path)

    logging.info(f"Masks written to {store_path}")


def mask_parse_args(args: list[str]) -> argparse.Namespace:
    """Parse command-line arguments for segmenting tracked crabs."""
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--boxes",
        type=str,
        required=True,
        help=(
            "Path to the trajectories zarr store, "
            "with the boxes used to prompt SAM2. "
            "Such a store is written by running `create-zarr-dataset`."
        ),
    )
    parser.add_argument(
        "--videos_dir",
        type=str,
        required=True,
        help=(
            "Directory holding the videos the boxes refer to. "
            "Each video file is a clip and is named as "
            "<video-id>-<clip-id>.mp4."
        ),
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="mask_output",
        help=(
            "Parent directory of the mask store, which is named "
            "<boxes-stem>_masks_<timestamp>.zarr. Ignored if "
            "--output_store is given. Default: mask_output."
        ),
    )
    parser.add_argument(
        "--output_store",
        type=str,
        default=None,
        help=(
            "Path to the mask zarr store to write. If provided, --output_dir "
            "is ignored. Use it when several runs must write "
            "into one store, e.g. a cluster array job with one task per "
            "video. Default: a timestamped store under --output_dir."
        ),
    )
    parser.add_argument(
        "--match",
        type=str,
        default="*",
        help=(
            "Glob selecting which video groups of the trajectories store to "
            "mask, e.g. '07.09.2023*'. Note that all clips of a selected "
            "video are included. Default: all."
        ),
    )
    parser.add_argument(
        "--mask_config_file",
        type=str,
        default=DEFAULT_MASK_CONFIG,
        help=(
            "Location of YAML config to control masking. "
            "Default: crabs/mask/config/mask_config.yaml."
        ),
    )
    parser.add_argument(
        "--accelerator",
        type=str,
        default="gpu",
        choices=["cpu", "mps", "gpu"],
        help="Accelerator for Pytorch. Default: gpu.",
    )
    parser.add_argument(
        "--zarr_mode_store",
        type=str,
        default="w-",
        choices=["w-", "a"],
        help=(
            "Mode to open the output zarr store (the whole .zarr directory) "
            "with, once at the start of the run. 'w-' fails if the store "
            "already exists; 'a' reuses an existing store, e.g. for a "
            "cluster array job where each task adds one video to the same "
            "store. Default: 'w-'."
        ),
    )
    parser.add_argument(
        "--zarr_mode_group",
        type=str,
        default="w-",
        choices=["w-", "w"],
        help=(
            "Mode to write each video's group inside the store with. "
            "'w-' fails if that video's group already exists; "
            "'w' overwrites only that video's group, leaving other videos in "
            "the store untouched (e.g. to re-mask one video). Default: 'w-'."
        ),
    )
    parser.add_argument(
        "--no_consolidate",
        action="store_true",
        help=(
            "Skip consolidating the output store's metadata at the end of "
            "the run. Use it when several runs write to the same store at "
            "once (e.g. a cluster array job) to avoid a race. Instead, "
            "consolidate the store once after all jobs finish. "
            "Default: consolidate."
        ),
    )
    return parser.parse_args(args)


def app_wrapper():
    """Wrap function to run the masking application."""
    logging.getLogger().setLevel(logging.INFO)

    torch.set_float32_matmul_precision("medium")

    main(mask_parse_args(sys.argv[1:]))


if __name__ == "__main__":
    app_wrapper()
