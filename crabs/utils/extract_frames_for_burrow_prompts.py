# %%
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import xarray as xr

# Hide attributes globally
xr.set_options(
    display_expand_attrs=False,
    display_style="html",
)

# %% %%%%%%%%%%%%%%
# Input data
# data_dir = Path().home() / "swc" / "project_crabs" / "data" / "CrabTracks"
crabs_zarr_dataset = "/Users/sofia/swc/CrabTracks/CrabTracks-slurm2478780-2478861-2489356.zarr"  # data_dir / "CrabTracks-slurm2478780-2478861-2489356.zarr"
data_vars_order = [
    "position",
    "shape",
    "confidence",
    "escape_state",
]

# TODO: add timestamp
output_dir = Path(__file__).parents[2] / "prompt_frames"
output_dir.mkdir(exist_ok=True)

percentile_th = 10

# %% %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
# Read dataset as an xarray datatree

dt = xr.open_datatree(
    crabs_zarr_dataset,
    engine="zarr",
    chunks={},
)

dt

# %%%%%%%%%%%%%%%%%%%%%%%
# Compute number of detections per video
counts_per_video_frame = {}
frames_per_clip_below_th = {}
frames_per_video_below_th = {}
for dt_video in dt.leaves:
    # Get video dataset
    ds_video = dt_video.ds
    ds_video.coords["clip_escape_first_frame_0idx"].load()

    # Get detections per clip and clip length
    count_per_clip_in_video = (
        (~ds_video.confidence.isnull()).sum(axis=-1).compute().values
    )  # (clip, max_frame_idx_all_clip)
    n_frames_per_clip = (
        ds_video.clip_last_frame_0idx - ds_video.clip_first_frame_0idx + 1
    ).values  # (clip,)

    # Concatenate counts per clip --> indices are frame idcs in video
    counts_per_video_frame[ds_video.video_id] = np.concatenate(
        [
            count_per_clip_in_video[i, :n]
            for i, n in enumerate(n_frames_per_clip)
        ]
    )

    # Compute count value such that X% of frame counts are below threshold
    count_th = np.percentile(
        counts_per_video_frame[ds_video.video_id], percentile_th
    )

    # Compute frame indices *per video* below threshold
    frames_per_video_below_th[ds_video.video_id] = np.where(
        counts_per_video_frame[ds_video.video_id] < count_th
    )[0]

    # Compute frame indices *per clip* below threshold
    # (that is, frame indices have origin at start of clip, NOT video)
    for i, clip_id in enumerate(ds_video.clip_id.values):
        n_frames_in_clip = n_frames_per_clip[i]
        frame_idcs_below_th = np.where(
            count_per_clip_in_video[i, :n_frames_in_clip] < count_th
        )[0]

        if len(frame_idcs_below_th) > 0:
            frames_per_clip_below_th[(ds_video.video_id, clip_id)] = (
                frame_idcs_below_th
            )

# %%%%%%%%%%%%%%%%%%%
# Check number of frames per video
# count_per_video for video_name='foo' should match ds_video.clip_last_frame_0idx + 1
# (more precisely: ds_video.clip_last_frame_0idx.isel(clip_id=-1).item()) +1
# Plot per video, mark escape frame

assert [
    len(counts_per_video_frame[ky])
    == dt[ky].ds.clip_last_frame_0idx.isel(clip_id=-1).item() + 1
    for ky in counts_per_video_frame
]


# %%%%%%%%%%%%%%%%%%%%%%
# Check frame indices to extract per video and clip are consistent

# # Loop thru videos
# for video_id in frames_per_video_below_th:
#     frames_from_video_start = frames_per_video_below_th[video_id]

#     # Only consider clips that have frames for extraction
#     list_of_relevant_clips = [
#         ky for ky in frames_per_clip_below_th if ky.split("_")[0] == video_id
#     ]
#     list_frame_idcs_per_clip = []
#     for video_clip_id in list_of_relevant_clips:
#         clip_id = video_clip_id.split("_")[1]
#         clip_start_frame = (
#             dt[video_id].clip_first_frame_0idx.sel(clip_id=clip_id).item()
#         )

#         list_frame_idcs_per_clip.append(
#             frames_per_clip_below_th[video_clip_id] + clip_start_frame
#         )

#     assert np.all(frames_from_video_start == np.concatenate(list_frame_idcs_per_clip))

# Loop thru clips
running_start_idx = 0
prev_video_id = list(frames_per_clip_below_th.keys())[0][0]
for video_clip_id, frames_per_clip in frames_per_clip_below_th.items():
    video_id, clip_id = video_clip_id
    clip_start_frame = (
        dt[video_id].clip_first_frame_0idx.sel(clip_id=clip_id).item()
    )
    n_extracted_frames = len(frames_per_clip)

    # Determine starting index in frames per video array
    # (reset to zero if video changes)
    start_idx = running_start_idx if video_id == prev_video_id else 0

    # Compare to the relevant section in the frames per video array
    assert np.all(
        frames_per_clip + clip_start_frame
        == frames_per_video_below_th[video_id][
            start_idx : start_idx + n_extracted_frames
        ]
    )

    # Update video running variables
    running_start_idx = start_idx + n_extracted_frames
    prev_video_id = video_id


# %%%%%%%%%%%%%%%%%%%%%%%%%
# Plot
# TODO: save as plotly
n_videos = len(counts_per_video_frame)
n_cols = 3
n_rows = int(np.ceil(n_videos / n_cols))
fig, axs = plt.subplots(n_rows, n_cols, figsize=(15, 3 * n_rows), sharex=False)
axs_flat = axs.flatten()

map_escape_type_to_style = {
    "triggered": ("red", "solid"),
    "spontaneous": ("red", "dashed"),
    "tourists": ("green", "dotted"),
}

fps = dt["04.09.2023-01-Right"].fps

for ax, ky in zip(axs_flat, counts_per_video_frame, strict=False):
    video_frame_idcs = np.arange(counts_per_video_frame[ky].shape[0])

    ax.plot(video_frame_idcs / fps / 60, counts_per_video_frame[ky])
    ax.set_title(ky, fontsize=9)
    ax.set_xlabel("time (min)")
    ax.set_ylabel("n detections")

    # common y axis for all
    ax.set_ylim([0, 120])

    # add escape frames as vertical lines
    escapes_video_frame_idcs = dt[ky].clip_escape_first_frame_0idx.values
    escape_type = dt[ky].clip_escape_type.values

    for x_val, esc_type in zip(
        escapes_video_frame_idcs, escape_type, strict=True
    ):
        ax.axvline(
            x=x_val / fps / 60,
            color=map_escape_type_to_style[esc_type][0],
            linestyle=map_escape_type_to_style[esc_type][1],
            alpha=0.5,
        )

    # add frames per clip below threshold

fig.tight_layout()

print(
    f"Min n detections per frame: {min([min(val) for val in counts_per_video_frame.values()])}"
)
print(
    f"Max n detections per frame: {max([max(val) for val in counts_per_video_frame.values()])}"
)

# %%%%%%%%%%%%%%%%%%%%%%%%%%%%%
# Save frames to extract as csv
# per clip and per video


# # Export frame indices relative to video start
# rows_per_video = [
#     {"video_id": video_id, "frame_idx_in_video": int(f)}
#     for video_id, frame_idcs in frames_per_video_below_th.items()
#     for f in frame_idcs
# ]
# df_per_video = pd.DataFrame(rows_per_video)
# df_per_video.to_csv(output_dir / "frames_per_video.csv", index=False)

# Export frame indices relative to clip start (and to video start for convenience)
rows_per_clip = []
for video_clip_id, frame_idcs in frames_per_clip_below_th.items():
    video_id, clip_id = video_clip_id
    clip_start = dt[video_id].clip_first_frame_0idx.sel(clip_id=clip_id).item()
    for f in frame_idcs:
        rows_per_clip.append(
            {
                "loop_clip_name": f"{video_id}-{clip_id}.mp4",
                "frame_0idx_in_clip": int(f),
                "video_name": f"{video_id}.mov",
                "frame_0idx_in_video": int(f) + clip_start,
            }
        )
df_per_clip = pd.DataFrame(rows_per_clip)
df_per_clip.to_csv(output_dir / "frames_per_clip.csv", index=False)

# %%
