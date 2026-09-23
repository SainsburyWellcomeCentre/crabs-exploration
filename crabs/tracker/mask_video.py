"""Mask tracked crabs in a video with SAM2."""

from pathlib import Path

import cv2
import dask.array as da
import numpy as np
import xarray as xr
import yaml  # type: ignore
import zarr
from zarr.codecs import BloscCodec, ZstdCodec

from crabs.tracker.utils.io import (
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
