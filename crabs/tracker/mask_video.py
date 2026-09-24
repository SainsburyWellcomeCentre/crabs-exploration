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
import yaml  # type: ignore
import zarr
from zarr.codecs import BloscCodec, ZstdCodec

from crabs.tracker.utils.boxes_from_zarr import read_tracked_bboxes_from_zarr
from crabs.tracker.utils.io import (
    get_video_parameters,
    open_video,
    parse_video_frame_reading_error_and_log,
)

DEFAULT_MASK_CONFIG = str(
    Path(__file__).parent / "config" / "mask_config.yaml"
)

# 0 is background in a label image, so crab labels start at 1
BACKGROUND_LABEL = 0


def load_mask_config(config_file: str) -> dict:
    """Load yaml file that contains the masking config parameters."""
    with open(config_file) as f:
        return yaml.safe_load(f)


def accelerator_to_device(accelerator: str) -> str:
    """Map a Pytorch accelerator name to a torch device name."""
    if accelerator == "gpu":
        return "cuda"
    return accelerator


def load_sam2_predictor(model_id: str, device: str):
    """Import sam2 lazily and return a SAM2 image predictor."""
    try:
        from sam2.sam2_image_predictor import SAM2ImagePredictor
    except ImportError as e:
        raise ImportError(
            "mask-tracked-crabs needs SAM2. Install it with:\n"
            "  uv sync --group masks\n"
            "or, in a conda env with torch already installed:\n"
            "  pip install --no-build-isolation "
            '"sam-2 @ git+https://github.com/facebookresearch/sam2.git"'
        ) from e
    return SAM2ImagePredictor.from_pretrained(model_id, device=device)


def create_mask_store(
    store_path: str | Path,
    video_id: str,
    clip_ids: list[str],
    n_frames: int,
    individuals: list[str],
    image_shape: tuple[int, int],
    metadata_dict: dict,
    shard_n_frames: int | None,
    zarr_mode_group: str,
) -> zarr.Array:
    """Write the template for one video group and return its labels array.

    The dataset holds one label image per frame --- a single integer array
    in which each crab's pixels carry that crab's own pixel value --- plus
    `label_of`, the individual to pixel value mapping. It has no
    `individual` dimension: that lives on `label_of`.

    The template is written with `compute=False`, so it puts metadata and
    coordinates on disk but no data. Each clip then fills its own region of
    the returned zarr array.

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
    shard_n_frames : int | None
        Number of frames per shard file. None disables sharding.
    zarr_mode_group : str
        Mode to write the zarr group with.

    Returns
    -------
    zarr.Array
        The `labels` array of the group just written.

    """
    height, width = image_shape
    chunks = (1, 1, height, width)
    shards = (1, shard_n_frames, height, width) if shard_n_frames else None

    ds = xr.Dataset(
        {
            "labels": (
                ("clip_id", "time", "img_h", "img_w"),
                # the dask chunks are only there to keep `to_zarr` lazy, but
                # they have to align with the unit zarr writes, i.e. the shard
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

    encoding: dict = {
        "labels": {
            "chunks": chunks,
            "compressors": [
                BloscCodec(cname="zstd", clevel=9, shuffle="bitshuffle")
            ],
        },
        # bitshuffle measured worse than plain zstd on this small,
        # mostly-zero integer array: codec per array, not per store
        "label_of": {"compressors": [ZstdCodec(level=9)]},
    }
    if shards:
        encoding["labels"]["shards"] = shards

    ds.to_zarr(
        store_path,
        group=video_id,
        mode=zarr_mode_group,
        encoding=encoding,
        compute=False,
    )

    return zarr.open_group(store_path)[f"{video_id}/labels"]


def predict_and_flatten_masks_into(
    predictor,
    frame_bgr: np.ndarray,
    boxes: np.ndarray,
    values: np.ndarray,
    label_frame: np.ndarray,
    max_prompts_per_batch: int,
) -> None:
    """Run SAM2 on one frame, flattened into the caller's label image.

    `values` are the pixel values for these boxes, read from `label_of`.
    They are the numbers written into `label_frame`, not positions along
    any axis.

    Flattening into `label_frame` in place, rather than returning one mask
    per prompt, is what releases a prompt chunk's full-frame float masks
    before the next chunk is predicted.

    The prompts are sorted by box area before chunking, because a chunk is
    painted before the next is predicted: without the sort, a large crab in
    a later chunk would overwrite a small crab from an earlier one, which
    is the opposite of the occlusion policy.
    """
    predictor.set_image(cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB))

    # largest box first, so chunk order agrees with the occlusion policy
    areas = (boxes[:, 2] - boxes[:, 0]) * (boxes[:, 3] - boxes[:, 1])
    order = np.argsort(-areas)
    boxes, values = boxes[order], values[order]

    for start in range(0, len(boxes), max_prompts_per_batch):
        box_chunk = boxes[start : start + max_prompts_per_batch]
        value_chunk = values[start : start + max_prompts_per_batch]

        masks, _scores, _low_res = predictor.predict(
            box=box_chunk, multimask_output=False
        )
        # predict() squeezes: (N, 1, H, W) for N>1 but (1, H, W) for N==1
        masks = masks.reshape(len(box_chunk), *masks.shape[-2:]).astype(bool)

        # smallest_wins: paint largest first, so where two masks overlap
        # the smaller crab keeps the contested pixels
        mask_areas = masks.reshape(len(box_chunk), -1).sum(axis=1)
        for i in np.argsort(-mask_areas):
            label_frame[masks[i]] = value_chunk[i]


def write_clip_masks_to_store(
    video_path: str,
    tracked_bboxes_dict: dict,
    labels_array: zarr.Array,
    clip_index: int,
    label_of: xr.DataArray,
    predictor,
    max_prompts_per_batch: int,
    shard_n_frames: int | None,
) -> int:
    """Mask one clip into `labels_array[clip_index]`.

    Frames are buffered a shard at a time and written whole, so zarr never
    has to read-modify-write a shard file. With sharding disabled the
    buffer is one frame, which is one chunk.

    Parameters
    ----------
    video_path : str
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
    shard_n_frames : int | None
        Number of frames per shard file. None disables sharding.

    Returns
    -------
    int
        The number of frames read from the clip video.

    """
    total_n_frames, height, width = labels_array.shape[1:]

    # a KeyError from this lookup means the `individual` coordinate and the
    # boxes dict disagree: loud, rather than a silently mislabelled mask
    value_of = {
        str(name): int(value)
        for name, value in zip(
            label_of.individual.values, label_of.values, strict=True
        )
    }

    # one shard's worth of label frames, allocated once per clip
    buffer_n_frames = shard_n_frames or 1
    buffer = np.zeros((buffer_n_frames, height, width), dtype=np.uint16)

    input_video_object = open_video(video_path)
    frame_idx = 0
    while input_video_object.isOpened():
        ret, frame = input_video_object.read()
        if not ret:
            parse_video_frame_reading_error_and_log(frame_idx, total_n_frames)
            break

        k = frame_idx % buffer_n_frames
        buffer[k] = BACKGROUND_LABEL

        # .get, not [...]: there may be no key for a frame with no boxes
        frame_data = tracked_bboxes_dict.get(frame_idx)
        if frame_data is not None and len(frame_data["tracked_boxes"]) > 0:
            predict_and_flatten_masks_into(
                predictor,
                frame,
                frame_data["tracked_boxes"],
                np.array([value_of[str(i)] for i in frame_data["ids"]]),
                buffer[k],
                max_prompts_per_batch,
            )

        # flush a whole shard at a time: never a partial-shard write
        if k == buffer_n_frames - 1:
            labels_array[clip_index, frame_idx - k : frame_idx + 1] = buffer
        frame_idx += 1

    # the tail: whatever is left of a part-filled shard
    k = frame_idx % buffer_n_frames
    if k:
        labels_array[clip_index, frame_idx - k : frame_idx] = buffer[:k]

    input_video_object.release()
    return frame_idx


def _resolve_clip_videos(
    videos_dir: str, video_id: str, clip_ids: list[str]
) -> list[Path]:
    """Map a video group's clips to the clip videos they refer to.

    `extract-loop-clips` writes one .mp4 per clip, which is the video the
    store's clip-local time axis refers to. The name is only ever
    formatted here, never parsed.
    """
    clip_videos = [
        Path(videos_dir) / f"{video_id}-{clip_id}.mp4" for clip_id in clip_ids
    ]
    for clip_video in clip_videos:
        if not clip_video.exists():
            raise FileNotFoundError(f"Clip video not found: {clip_video}")
    return clip_videos


def _video_group_image_shape_and_n_frames(
    clip_videos: list[Path], video_id: str
) -> tuple[tuple[int, int], int]:
    """Read the frame size and the longest clip of a video group.

    One group holds a single frame size, so a group whose clips disagree
    fails naming them, rather than writing truncated masks.
    """
    video_params = [get_video_parameters(str(p)) for p in clip_videos]
    image_shapes = {
        (p["frame_height"], p["frame_width"]) for p in video_params
    }
    if len(image_shapes) > 1:
        raise ValueError(
            f"{video_id}: its clips differ in frame size {image_shapes}, "
            "but one mask store group holds a single frame size."
        )
    return image_shapes.pop(), max(p["total_frames"] for p in video_params)


def main(args: argparse.Namespace) -> None:
    """Mask the tracked crabs of a trajectories store with SAM2."""
    boxes_path = Path(args.boxes)
    if boxes_path.suffix != ".zarr":
        raise ValueError(
            f"--boxes must be a trajectories zarr store (.zarr), but "
            f"'{boxes_path.name}' has suffix '{boxes_path.suffix}'."
        )

    mask_config = load_mask_config(args.mask_config_file)
    sam2_model_id = mask_config.get(
        "sam2_model_id", "facebook/sam2.1-hiera-base-plus"
    )
    max_prompts_per_batch = mask_config.get("max_prompts_per_batch", 32)
    shard_n_frames = mask_config.get("shard_n_frames", 32)
    occlusion_policy = mask_config.get("occlusion_policy", "smallest_wins")
    if occlusion_policy != "smallest_wins":
        raise ValueError(
            f"Unknown occlusion_policy '{occlusion_policy}'. The only policy "
            "implemented is 'smallest_wins'."
        )

    datatree = xr.open_datatree(args.boxes, engine="zarr", chunks={})
    # a pattern that matches nothing leaves only the root node behind
    video_nodes = [
        node for node in datatree.match(args.match).leaves if node.path != "/"
    ]
    if not video_nodes:
        raise ValueError(
            f"--match '{args.match}' selected no video group of {args.boxes}."
        )

    # Resolve every clip video before SAM2 is loaded, so a mis-pointed
    # --videos costs a second rather than a model download
    map_video_to_clips = {}
    for node in video_nodes:
        clip_ids = [str(c) for c in node.clip_id.values]
        map_video_to_clips[node.name] = (
            clip_ids,
            _resolve_clip_videos(args.videos, node.name, clip_ids),
        )

    device = accelerator_to_device(args.accelerator)
    predictor = load_sam2_predictor(sam2_model_id, device)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    if args.output_store:
        store_path = Path(args.output_store)
    else:
        # the default name is timestamped, so re-masking does not collide
        store_path = (
            Path(args.output_dir) / f"{boxes_path.stem}_masks_{timestamp}.zarr"
        )
    store_path.parent.mkdir(parents=True, exist_ok=True)
    zarr.open_group(store_path, mode=args.zarr_mode_store)

    for node in video_nodes:
        video_id = node.name
        ds_video = node.to_dataset()
        clip_ids, clip_videos = map_video_to_clips[video_id]

        # Every dimension is known from the trajectories store and the
        # video headers, so the template needs no two-pass temp store
        image_shape, n_frames = _video_group_image_shape_and_n_frames(
            clip_videos, video_id
        )

        labels_array = create_mask_store(
            store_path=store_path,
            video_id=video_id,
            clip_ids=clip_ids,
            n_frames=n_frames,
            # copied from the trajectories store, never derived
            individuals=[str(i) for i in ds_video.individual.values],
            image_shape=image_shape,
            metadata_dict={
                "timestamp": timestamp,
                "sam2_model": sam2_model_id,
                "device": device,
                "source_video": [str(p) for p in clip_videos],
                "boxes": {
                    "file": str(boxes_path),
                    "source": "trajectories_zarr",
                    "ids": "trajectories_store_individual",
                },
                "background_label": BACKGROUND_LABEL,
                "occlusion_policy": occlusion_policy,
                "prompt_type": "bounding_box",
            },
            shard_n_frames=shard_n_frames,
            zarr_mode_group=args.zarr_mode_group,
        )
        label_of = xr.open_datatree(store_path, engine="zarr")[
            video_id
        ].label_of

        for clip_index, clip_id in enumerate(clip_ids):
            logging.info(f"Masking {video_id}/{clip_id}")
            tracked_bboxes_dict, _ = read_tracked_bboxes_from_zarr(
                ds_video, clip_id
            )
            if not tracked_bboxes_dict:
                logging.info(
                    f"{video_id}/{clip_id}: no tracked boxes, skipping"
                )
                continue
            write_clip_masks_to_store(
                video_path=str(clip_videos[clip_index]),
                tracked_bboxes_dict=tracked_bboxes_dict,
                labels_array=labels_array,
                clip_index=clip_index,
                label_of=label_of,
                predictor=predictor,
                max_prompts_per_batch=max_prompts_per_batch,
                shard_n_frames=shard_n_frames,
            )

    logging.info(f"Masks written to {store_path}")


def mask_parse_args(args: list[str]) -> argparse.Namespace:
    """Parse command-line arguments for masking tracked crabs."""
    parser = argparse.ArgumentParser(
        # the default formatter re-wraps the epilog to terminal width and
        # breaks the command name across two lines, on its hyphens
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=(
            "To run detection, tracking and masking in one command, "
            "use detect-and-track-mask."
        ),
    )
    parser.add_argument(
        "--boxes",
        type=str,
        required=True,
        help=(
            "Location of the tracked boxes to prompt SAM2 with: a "
            "trajectories zarr store written by create-zarr-dataset. "
            "The suffix selects how it is read."
        ),
    )
    parser.add_argument(
        "--videos",
        type=str,
        required=True,
        help=(
            "Directory holding the clip videos the boxes refer to, named "
            "<video-id>-<clip-id>.mp4."
        ),
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="mask_output",
        help=(
            "Directory the mask store is written into. The store name "
            "carries a timestamp, so runs do not collide. "
            "Default: mask_output."
        ),
    )
    parser.add_argument(
        "--output_store",
        type=str,
        default=None,
        help=(
            "Path to the mask zarr store to write, naming it exactly and "
            "ignoring --output_dir. Use it when several runs must write "
            "into one store, e.g. a cluster array job with one task per "
            "video. Default: a timestamped store under --output_dir."
        ),
    )
    parser.add_argument(
        "--match",
        type=str,
        default="*",
        help=(
            "Glob selecting which video groups of the store to mask, e.g. "
            "'07.09.2023*'. Every clip of a selected video is masked. "
            "Default: all."
        ),
    )
    parser.add_argument(
        "--mask_config_file",
        type=str,
        default=DEFAULT_MASK_CONFIG,
        help=(
            "Location of YAML config to control masking. "
            "Default: crabs/tracker/config/mask_config.yaml."
        ),
    )
    parser.add_argument(
        "--accelerator",
        type=str,
        default="gpu",
        help=(
            "Accelerator for Pytorch. "
            "Valid inputs are: cpu, mps, or gpu. Default: gpu."
        ),
    )
    parser.add_argument(
        "--zarr_mode_store",
        type=str,
        default="w-",
        help=(
            "Mode to open the output zarr store with. "
            "Default: 'w-' (will fail if store exists). "
            "Use 'a' to append, e.g. from a cluster array job with one "
            "job per video."
        ),
    )
    parser.add_argument(
        "--zarr_mode_group",
        type=str,
        default="w-",
        help=(
            "Mode to write each video's zarr group with. "
            "Default: 'w-' (will fail if group exists)."
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
