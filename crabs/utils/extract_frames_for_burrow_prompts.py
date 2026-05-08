# %%
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import xarray as xr
from plotly.subplots import make_subplots

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
timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
output_dir = Path(__file__).parents[2] / f"prompt_frames_{timestamp}"
output_dir.mkdir(exist_ok=True)

percentile_th = 5

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
    video_id = ds_video.video_id

    # Get detections per clip and clip length
    count_per_clip_in_video = (
        (~ds_video.confidence.isnull()).sum(axis=-1).compute().values
    )  # (clip, max_frame_idx_all_clip)
    n_frames_per_clip = (
        ds_video.clip_last_frame_0idx - ds_video.clip_first_frame_0idx + 1
    ).values  # (clip,)

    # Get counts per video frame by concatenating counts per clip
    counts_video = np.concatenate(
        [
            count_per_clip_in_video[i, :n]
            for i, n in enumerate(n_frames_per_clip)
        ]
    )
    counts_per_video_frame[video_id] = counts_video

    # Compute count value such that X% of frame counts per video
    # are below threshold
    count_th = np.percentile(counts_video, percentile_th)
    below_th_mask = counts_video < count_th

    # Compute frame indices *per video* below threshold
    frames_per_video_below_th[video_id] = np.where(below_th_mask)[0]

    # Compute frame indices *per clip* below threshold
    clip_boundaries = np.concatenate([[0], np.cumsum(n_frames_per_clip)])
    for i, clip_id in enumerate(ds_video.clip_id.values):
        below_th_mask_clip = below_th_mask[
            clip_boundaries[i] : clip_boundaries[i + 1]
        ]
        clip_idcs = np.where(below_th_mask_clip)[0]
        if len(clip_idcs) > 0:
            frames_per_clip_below_th[(video_id, clip_id)] = clip_idcs

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

# Loop thru videos
for video_id in frames_per_video_below_th:
    frames_from_video_start = frames_per_video_below_th[video_id]

    # Only consider clips that have frames for extraction
    list_of_relevant_clips = [
        ky for ky in frames_per_clip_below_th if ky[0] == video_id
    ]
    list_frame_idcs_per_clip = []
    for video_clip_id in list_of_relevant_clips:
        _, clip_id = video_clip_id
        clip_start_frame = (
            dt[video_id].clip_first_frame_0idx.sel(clip_id=clip_id).item()
        )

        list_frame_idcs_per_clip.append(
            frames_per_clip_below_th[video_clip_id] + clip_start_frame
        )

    assert np.all(
        frames_from_video_start == np.concatenate(list_frame_idcs_per_clip)
    )


# %%%%%%%%%%%%%%%%%%%%%%%%%%%%%
# Save frames to extract as csv
# per clip and per video

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
csv_path = output_dir / "frames_per_clip.csv"
with open(csv_path, "w") as f:
    # Add percentile used as metadata
    # we can use pd.read_csv(path, comment="#") to skip it when reading
    f.write(f"# percentile_th={percentile_th}\n")
    df_per_clip.to_csv(f, index=False)

# %%%%%%%%%%%%%%%%%%%%%%%%%
# Plot count per frame, escape frames and selected frames for extraction

map_escape_type_to_plotly_style = {
    "triggered": ("red", "solid"),
    "spontaneous": ("red", "dash"),
    "tourists": ("green", "solid"),
}

n_videos = len(counts_per_video_frame)
n_cols = 3
n_rows = int(np.ceil(n_videos / n_cols))

fig_plotly = make_subplots(
    rows=n_rows,
    cols=n_cols,
    subplot_titles=list(counts_per_video_frame.keys()),
    horizontal_spacing=0.04,
    vertical_spacing=0.025,
)

fps = dt["04.09.2023-01-Right"].ds.fps

# Track which escape types have been added to legend, to deduplicate
escape_types_in_legend = set()

for i, ky in enumerate(counts_per_video_frame):
    row = i // n_cols + 1
    col = i % n_cols + 1

    video_frame_idcs = np.arange(counts_per_video_frame[ky].shape[0])
    time_min = video_frame_idcs / fps / 60

    # plot counts per frame
    fig_plotly.add_trace(
        go.Scattergl(
            x=time_min,
            y=counts_per_video_frame[ky],
            mode="lines",
            line=dict(color="#1f77b4", width=1),
            name=ky,
            showlegend=False,
            hovertemplate=(
                "t=%{x:.2f} min<br>n_detections=%{y}<extra></extra>"
            ),
        ),
        row=row,
        col=col,
    )

    # add scatter markers for the selected frames (below threshold)
    selected_frame_idcs = frames_per_video_below_th[ky]
    show_legend_selected = "selected_frames" not in escape_types_in_legend
    escape_types_in_legend.add("selected_frames")
    fig_plotly.add_trace(
        go.Scattergl(
            x=selected_frame_idcs / fps / 60,
            y=counts_per_video_frame[ky][selected_frame_idcs],
            mode="markers",
            marker=dict(
                symbol="circle",
                size=2.5,
                color="orange",
                # line=dict(color="black", width=0.5),
            ),
            name=f"selected frames (<{percentile_th}th percentile)",
            legendgroup="selected_frames",
            showlegend=show_legend_selected,
            hovertemplate=(
                "t=%{x:.2f} min<br>n_detections=%{y}<extra></extra>"
            ),
        ),
        row=row,
        col=col,
    )

    # add escape frames as vertical lines
    escapes_video_frame_idcs = dt[ky].clip_escape_first_frame_0idx.values
    escape_type = dt[ky].clip_escape_type.values

    for x_val, esc_type in zip(
        escapes_video_frame_idcs, escape_type, strict=True
    ):
        color, dash = map_escape_type_to_plotly_style[esc_type]
        x_min = float(x_val) / fps / 60
        show_legend = esc_type not in escape_types_in_legend
        escape_types_in_legend.add(esc_type)
        fig_plotly.add_trace(
            go.Scattergl(
                x=[x_min, x_min],
                y=[0, 120],
                mode="lines",
                line=dict(color=color, dash=dash, width=1.5),
                opacity=0.65,
                name=esc_type,
                legendgroup=esc_type,
                showlegend=show_legend,
                hoverinfo="skip",
            ),
            row=row,
            col=col,
        )

    # set x and y axes labels
    is_bottom_row = row == n_rows or (
        row == n_rows - 1 and i + n_cols >= n_videos
    )
    fig_plotly.update_xaxes(
        title_text="time (min)" if is_bottom_row else None,
        row=row,
        col=col,
    )
    fig_plotly.update_yaxes(
        title_text="n detections" if col == 1 else None,
        range=[0, 120],
        row=row,
        col=col,
    )

for annotation in fig_plotly.layout.annotations:
    annotation.font.size = 12

fig_plotly.update_layout(
    height=180 * n_rows,
    width=1500,
    plot_bgcolor="white",
    paper_bgcolor="white",
    legend=dict(x=1.02, y=1, yanchor="top", xanchor="left"),
    margin=dict(l=50, r=50, t=30, b=30),
)
fig_plotly.update_xaxes(
    showgrid=True,
    gridcolor="lightgrey",
    linecolor="black",
    mirror=True,
    ticks="outside",
    zeroline=False,
)
fig_plotly.update_yaxes(
    showgrid=True,
    gridcolor="lightgrey",
    linecolor="black",
    mirror=True,
    ticks="outside",
    zeroline=False,
)


fig_plotly.write_html(str(output_dir / "n_detections_per_video.html"))

print(
    f"Min n detections per frame: {min([min(val) for val in counts_per_video_frame.values()])}"
)
print(
    f"Max n detections per frame: {max([max(val) for val in counts_per_video_frame.values()])}"
)

# %%
