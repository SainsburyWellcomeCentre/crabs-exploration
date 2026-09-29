"""Time each stage of mask-tracked-crabs on the first frames of one clip.

It mirrors `predict_and_flatten_masks_into`, but splits SAM2's `predict`
into its GPU part and its GPU-to-CPU copy, and also times painting on the
GPU, so the per-frame cost can be attributed before optimising it.

Example, on a GPU node, in an environment with crabs[masks] installed:

    python scripts/profile_mask_timing.py \
        --boxes /ceph/zoo/processed/CrabField/ramalhete_2023/CrabTracks/CrabTracks-slurm3644250.zarr \
        --videos /ceph/zoo/processed/CrabField/ramalhete_2023/Loops \
        --video_id 04.09.2023-01-Right --clip_id Loop00
"""

import argparse
import logging
import time
from contextlib import contextmanager

import cv2
import numpy as np
import torch
import xarray as xr

from crabs.mask.mask_video import load_sam2_predictor
from crabs.mask.utils.boxes_from_zarr import read_tracked_bboxes_from_zarr


@contextmanager
def timer(stage: str, frame_times: dict):
    """Add a stage's wall-clock time to `frame_times`, syncing CUDA."""
    torch.cuda.synchronize()
    start = time.perf_counter()
    yield
    torch.cuda.synchronize()
    frame_times[stage] = frame_times.get(stage, 0.0) + (
        time.perf_counter() - start
    )


def profile_frame(predictor, frame_bgr, boxes, max_prompts_per_batch):
    """Run one frame through every stage and return seconds per stage."""
    t: dict[str, float] = {}
    height, width = frame_bgr.shape[:2]
    label_frame = np.zeros((height, width), dtype=np.uint16)
    label_frame_gpu = torch.zeros(
        (height, width), dtype=torch.int32, device="cuda"
    )
    # stand-in pixel values: only the cost matters here
    values = np.arange(1, len(boxes) + 1)

    with timer("1_cvtColor", t):
        frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
    with timer("2_set_image (encoder)", t):
        predictor.set_image(frame_rgb)

    areas = (boxes[:, 2] - boxes[:, 0]) * (boxes[:, 3] - boxes[:, 1])
    order = np.argsort(-areas)
    boxes, values = boxes[order], values[order]

    for start in range(0, len(boxes), max_prompts_per_batch):
        box_chunk = boxes[start : start + max_prompts_per_batch]
        value_chunk = values[start : start + max_prompts_per_batch]

        # the two halves of SAM2ImagePredictor.predict, timed separately
        with timer("3_predict: decoder + upsample (GPU)", t):
            _, _, _, unnorm_box = predictor._prep_prompts(
                None, None, box_chunk, None, True
            )
            masks_gpu, _, _ = predictor._predict(
                None, None, unnorm_box, None, multimask_output=False
            )
            masks_gpu = masks_gpu.reshape(len(box_chunk), height, width)
        with timer("4_predict: .float().cpu() copy", t):
            masks = masks_gpu.float().cpu().numpy()

        # what mask_video.py does with the copied masks
        with timer("5_astype(bool)", t):
            masks = masks.astype(bool)
        with timer("6_mask area sum (CPU)", t):
            mask_areas = masks.reshape(len(box_chunk), -1).sum(axis=1)
        with timer("7_paint (CPU)", t):
            for i in np.argsort(-mask_areas):
                label_frame[masks[i]] = value_chunk[i]

        # the alternative: flatten on the GPU, never copy the masks
        with timer("8_alt: area sum + paint (GPU)", t):
            gpu_areas = masks_gpu.flatten(1).sum(dim=1)
            for i in torch.argsort(-gpu_areas).tolist():
                label_frame_gpu.masked_fill_(masks_gpu[i], int(value_chunk[i]))
        del masks, masks_gpu

    with timer("9_alt: label frame .cpu() copy", t):
        label_frame_gpu.to(torch.uint16).cpu().numpy()
    return t


def main():
    """Profile the first frames of one clip."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--boxes", required=True)
    parser.add_argument("--videos", required=True)
    parser.add_argument("--video_id", required=True)
    parser.add_argument("--clip_id", required=True)
    parser.add_argument("--n_frames", type=int, default=50)
    parser.add_argument("--warmup", type=int, default=5)
    parser.add_argument(
        "--model_id", default="facebook/sam2.1-hiera-base-plus"
    )
    parser.add_argument("--max_prompts_per_batch", type=int, default=32)
    args = parser.parse_args()

    torch.set_float32_matmul_precision("medium")
    # SAM2 logs three INFO lines on every set_image call
    logging.getLogger().setLevel(logging.WARNING)
    print(torch.cuda.get_device_name(0), "| CPUs:", torch.get_num_threads())

    ds_video = xr.open_datatree(args.boxes, engine="zarr", chunks={})[
        args.video_id
    ].to_dataset()
    boxes_dict, _ = read_tracked_bboxes_from_zarr(ds_video, args.clip_id)
    predictor = load_sam2_predictor(args.model_id, "cuda")

    cap = cv2.VideoCapture(f"{args.videos}/{args.video_id}-{args.clip_id}.mp4")
    rows, n_boxes = [], []
    for frame_idx in range(args.warmup + args.n_frames):
        start = time.perf_counter()
        ret, frame = cap.read()
        decode_s = time.perf_counter() - start
        if not ret:
            break
        frame_data = boxes_dict.get(frame_idx)
        if frame_data is None or len(frame_data["tracked_boxes"]) == 0:
            continue
        t = profile_frame(
            predictor,
            frame,
            frame_data["tracked_boxes"],
            args.max_prompts_per_batch,
        )
        if frame_idx >= args.warmup:
            t["0_video decode"] = decode_s
            rows.append(t)
            n_boxes.append(len(frame_data["tracked_boxes"]))
    cap.release()

    print(
        f"\n{len(rows)} frames of {frame.shape[1]}x{frame.shape[0]}, "
        f"{np.mean(n_boxes):.0f} boxes/frame on average\n"
    )
    stages = sorted(rows[0])
    # stages are numbered: 0-7 is today's pipeline, 0-3 plus 8-9 the
    # GPU-paint alternative
    totals = {
        "current total": [s for s in stages if s[0] in "01234567"],
        "GPU-paint total": [s for s in stages if s[0] in "012389"],
    }
    print(f"{'stage':42s} {'ms/frame':>10s}")
    for stage in stages:
        print(f"{stage:42s} {1000 * np.mean([r[stage] for r in rows]):10.1f}")
    for name, group in totals.items():
        total = np.mean([sum(r[s] for s in group) for r in rows])
        print(f"{name:42s} {1000 * total:10.1f}")


if __name__ == "__main__":
    main()
