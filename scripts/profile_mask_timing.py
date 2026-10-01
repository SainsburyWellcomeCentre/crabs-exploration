"""Estimate the speed-up of the mask store's compressor change on the cluster.

It runs `predict_and_flatten_masks_into` on the first frames of one clip,
as `mask-tracked-crabs` does, then writes the label frames shard by shard
twice: with the old compressor (Blosc zstd, clevel=9, bitshuffle) and with
zarr's default, which the mask store now uses. The shards are written
under --scratch_dir, so run it from the filesystem the real jobs write
to. It prints the time per frame of each, and projects it to whole videos.

Example, on a GPU node, in an environment with crabs[masks] installed:

    DATA=/ceph/zoo/processed/CrabField/ramalhete_2023
    cd /ceph/zoo/users/sminano
    python /path/to/crabs-exploration/scripts/profile_mask_timing.py \
        --boxes $DATA/CrabTracks/CrabTracks-slurm3644250.zarr \
        --videos $DATA/Loops \
        --video_id 04.09.2023-01-Right --clip_id Loop00
"""

import argparse
import logging
import os
import shutil
import tempfile
import time
from pathlib import Path

import cv2
import numpy as np
import torch
import xarray as xr
import zarr
from zarr.codecs import BloscCodec

from crabs.mask.mask_video import (
    load_sam2_predictor,
    predict_and_flatten_masks_into,
)
from crabs.mask.utils.boxes_from_zarr import read_tracked_bboxes_from_zarr

COMPRESSORS = {
    "old: Blosc zstd clevel=9": [
        BloscCodec(cname="zstd", clevel=9, shuffle="bitshuffle")
    ],
    "new: zarr default": "auto",
}

# frame counts of the smallest, median and largest videos in the store
VIDEO_N_FRAMES = [70_000, 216_000, 330_000]


def log_memory(step: str) -> None:
    """Print current and peak host memory, flushed in case of an OOM kill."""
    with open("/proc/self/status") as f:
        status = dict(line.split(":", 1) for line in f)
    current_gb, peak_gb = (
        int(status[key].split()[0]) / 1e6 for key in ("VmRSS", "VmHWM")
    )
    print(
        f"[memory] {step}: {current_gb:.1f} GB now, {peak_gb:.1f} GB peak",
        flush=True,
    )


def time_shard_writes(
    label_frames: np.ndarray,
    compressors,
    shard_n_frames: int,
    scratch_dir: str,
) -> tuple[float, float]:
    """Write the frames shard by shard; return s/frame and bytes/frame."""
    n_frames, height, width = label_frames.shape
    store_dir = tempfile.mkdtemp(dir=scratch_dir, prefix="mask_codec_")
    try:
        # the same chunks and shards as the mask store, minus clip_id
        array = zarr.create_array(
            store_dir,
            shape=label_frames.shape,
            chunks=(1, height, width),
            shards=(shard_n_frames, height, width),
            dtype=np.uint16,
            compressors=compressors,
        )
        start = time.perf_counter()
        for first in range(0, n_frames, shard_n_frames):
            array[first : first + shard_n_frames] = label_frames[
                first : first + shard_n_frames
            ]
        seconds = time.perf_counter() - start
        n_bytes = sum(
            f.stat().st_size for f in Path(store_dir).rglob("*") if f.is_file()
        )
    finally:
        shutil.rmtree(store_dir)
    return seconds / n_frames, n_bytes / n_frames


def main():
    """Time the SAM2 pass and both shard writes on one clip."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--boxes", required=True)
    parser.add_argument("--videos", required=True)
    parser.add_argument("--video_id", required=True)
    parser.add_argument("--clip_id", required=True)
    parser.add_argument(
        "--n_frames",
        type=int,
        default=64,
        help="Frames to time, after the warm-up. Default: 64, two shards.",
    )
    parser.add_argument("--warmup", type=int, default=5)
    parser.add_argument(
        "--model_id", default="facebook/sam2.1-hiera-base-plus"
    )
    parser.add_argument("--max_prompts_per_batch", type=int, default=32)
    parser.add_argument("--shard_n_frames", type=int, default=32)
    parser.add_argument(
        "--scratch_dir",
        default=".",
        help="Where the test shards are written, then deleted. Default: .",
    )
    args = parser.parse_args()

    torch.set_float32_matmul_precision("medium")
    # SAM2 logs three INFO lines on every set_image call
    logging.getLogger().setLevel(logging.WARNING)
    print(
        f"{torch.cuda.get_device_name(0)} | "
        f"CPUs allocated: {len(os.sched_getaffinity(0))}"
    )

    ds_video = xr.open_datatree(args.boxes, engine="zarr", chunks={})[
        args.video_id
    ].to_dataset()
    boxes_dict, _ = read_tracked_bboxes_from_zarr(ds_video, args.clip_id)
    # the pixel values the mask store's label_of assigns
    value_of = {
        str(name): i + 1 for i, name in enumerate(ds_video.individual.values)
    }
    log_memory("boxes read")
    predictor = load_sam2_predictor(args.model_id, "cuda")
    log_memory("SAM2 loaded")

    cap = cv2.VideoCapture(f"{args.videos}/{args.video_id}-{args.clip_id}.mp4")
    log_memory("video opened")
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    label_frames = np.zeros((args.n_frames, height, width), dtype=np.uint16)
    decode_s, sam2_s, n_boxes = [], [], []
    for frame_idx in range(args.warmup + args.n_frames):
        start = time.perf_counter()
        ret, frame = cap.read()
        decoded = time.perf_counter()
        if not ret:
            raise ValueError(f"Clip ended at frame {frame_idx}")

        # warm-up frames are masked into a throwaway frame, and not timed
        k = frame_idx - args.warmup
        label_frame = (
            label_frames[k] if k >= 0 else np.zeros((height, width), np.uint16)
        )
        frame_data = boxes_dict.get(frame_idx)
        n = 0 if frame_data is None else len(frame_data["tracked_boxes"])
        if n > 0:
            predict_and_flatten_masks_into(
                predictor,
                frame,
                frame_data["tracked_boxes"],
                np.array([value_of[str(i)] for i in frame_data["ids"]]),
                label_frame,
                args.max_prompts_per_batch,
            )
        # it ends copying the label frame to the CPU, so the GPU is done
        if k >= 0:
            decode_s.append(decoded - start)
            sam2_s.append(time.perf_counter() - decoded)
            n_boxes.append(n)
        if frame_idx % 8 == 0:
            log_memory(f"frame {frame_idx}")
    cap.release()

    print(
        f"\n{args.n_frames} frames of {width}x{height}, "
        f"{np.mean(n_boxes):.0f} boxes/frame on average, "
        f"shards of {args.shard_n_frames} frames written to "
        f"{Path(args.scratch_dir).resolve()}\n"
    )
    per_frame = {
        "video decode": np.mean(decode_s),
        "SAM2 + paint on GPU": np.mean(sam2_s),
    }
    kb_per_frame = {}
    for name, compressors in COMPRESSORS.items():
        seconds, n_bytes = time_shard_writes(
            label_frames, compressors, args.shard_n_frames, args.scratch_dir
        )
        log_memory(f"written, {name}")
        per_frame[f"write, {name}"] = seconds
        kb_per_frame[name] = n_bytes / 1e3

    print(f"{'stage':36s} {'ms/frame':>10s} {'kB/frame':>10s}")
    for stage, seconds in per_frame.items():
        codec = stage.removeprefix("write, ")
        size = f"{kb_per_frame[codec]:10.0f}" if codec in kb_per_frame else ""
        print(f"{stage:36s} {1000 * seconds:10.1f} {size}")

    shared = per_frame["video decode"] + per_frame["SAM2 + paint on GPU"]
    totals = {
        name: shared + per_frame[f"write, {name}"] for name in COMPRESSORS
    }
    print()
    for name, total in totals.items():
        hours = ", ".join(
            f"{n // 1000}k frames: {n * total / 3600:5.1f} h"
            for n in VIDEO_N_FRAMES
        )
        print(f"total, {name:29s} {1000 * total:10.1f}   ({hours})")
    old, new = totals.values()
    print(f"\nspeed-up: {old / new:.1f}x")


if __name__ == "__main__":
    main()
