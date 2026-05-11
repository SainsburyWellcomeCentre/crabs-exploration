"""Extract frames for prompting SAM3 to detect burrows.

For every video in the input zarr store, we compute the number of crab
detections per frame (concatenating detections across clips), and select a
fixed fraction of frames with the lowest detection counts. These low-count
frames are good candidates for annotating burrows, since the scene is less
crowded.

The selected frame indices are written to a single CSV under a timestamped
output directory, with both per-clip and per-video frame indices. Optionally,
an interactive plotly HTML figure with the per-video detection counts and the
selected frames overlaid is also saved.

Output CSV columns (one row per selected frame):
    loop_clip_name,
    frame_0idx_in_clip,
    video_name,
    frame_0idx_in_video

A leading ``# frames_per_video_fraction=...`` comment row is written to
preserve the parameter used; it can be skipped with
``pd.read_csv(path, comment="#")``.

Usage (dependencies are auto-installed via uv):
* Default
    uv run extract_frames_for_burrow_prompts.py /path/to/store.zarr
    /path/to/out_dir
* Custom fraction
    uv run extract_frames_for_burrow_prompts.py /path/to/store.zarr
    /path/to/out_dir --frames-per-video-fraction 0.1
* Save HTML figure
    uv run extract_frames_for_burrow_prompts.py /path/to/store.zarr
    /path/to/out_dir --save-html-figure
"""

# /// script
# requires-python = ">=3.11"
# dependencies = [
#   "numpy>=2.0.0",
#   "pandas",
#   "xarray",
#   "dask",
#   "zarr",
#   "plotly",
# ]
# ///

import argparse
import sys
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import xarray as xr
from plotly.subplots import make_subplots


def _counts_per_video_frame(ds_video: xr.Dataset) -> np.ndarray:
    """Return the number of non-null detections per video frame.

    Concatenates per-clip non-null ``confidence`` counts (truncated to each
    clip's actual frame count) into a single array of length
    ``clip_last_frame_0idx[-1] + 1``.
    """
    count_per_clip_in_video = (
        (~ds_video.confidence.isnull()).sum(axis=-1).compute().values
    )
    n_frames_per_clip = (
        ds_video.clip_last_frame_0idx - ds_video.clip_first_frame_0idx + 1
    ).values
    return np.concatenate(
        [
            count_per_clip_in_video[i, :n]
            for i, n in enumerate(n_frames_per_clip)
        ]
    )


def _select_lowest_count_frame_idcs(
    counts_video: np.ndarray, n_to_extract: int
) -> np.ndarray:
    """Return indices of the ``n_to_extract`` frames with the lowest counts.

    Indices are sorted by count (ascending). Ties are broken by earliest
    frame index (stable sort).
    """
    bottom_idcs = np.argpartition(counts_video, n_to_extract)[:n_to_extract]
    return bottom_idcs[np.argsort(counts_video[bottom_idcs], kind="stable")]


def _video_idcs_to_per_clip_idcs(
    video_idcs: np.ndarray,
    n_frames_per_clip: np.ndarray,
    clip_ids: np.ndarray,
) -> dict[str, np.ndarray]:
    """Convert per-video frame indices to per-clip frame indices.

    Returns a dict mapping ``clip_id`` to the array of frame indices within
    that clip (preserving the input order). Clips with no selected frames are
    omitted.
    """
    clip_boundaries = np.concatenate([[0], np.cumsum(n_frames_per_clip)])
    clip_id_per_idx = (
        np.searchsorted(clip_boundaries, video_idcs, side="right") - 1
    )
    idcs_within_clip = video_idcs - clip_boundaries[clip_id_per_idx]

    result: dict[str, np.ndarray] = {}
    for i, clip_id in enumerate(clip_ids):
        mask = clip_id_per_idx == i
        if mask.any():
            result[clip_id] = idcs_within_clip[mask]
    return result


def _build_rows_per_clip(
    per_clip_idcs: dict[str, np.ndarray],
    video_id: str,
    ds_video: xr.Dataset,
) -> list[dict]:
    """Build CSV rows for one video from the per-clip selected indices."""
    rows = []
    for clip_id, frame_idcs in per_clip_idcs.items():
        clip_start = ds_video.clip_first_frame_0idx.sel(clip_id=clip_id).item()
        for f in frame_idcs:
            rows.append(
                {
                    "loop_clip_name": f"{video_id}-{clip_id}.mp4",
                    "frame_0idx_in_clip": int(f),
                    "video_name": f"{video_id}.mov",
                    "frame_0idx_in_video": int(f) + clip_start,
                }
            )
    return rows


def plot_n_detections_html(
    counts_per_video_frame: dict[str, np.ndarray],
    frames_per_video_to_extract: dict[str, np.ndarray],
    dt: xr.DataTree,
    frames_per_video_fraction: float,
    output_html_path: Path,
) -> None:
    """Plot per-video detection counts with selected frames and escapes.

    One subplot per video in a grid, showing detections per frame, markers
    at the selected (lowest-count) frames, and vertical lines at escape
    frames coloured by escape type.
    """
    map_escape_type_to_plotly_style = {
        "triggered": ("red", "solid"),
        "spontaneous": ("red", "dash"),
        "tourists": ("green", "solid"),
    }

    n_videos = len(counts_per_video_frame)
    n_cols = 3
    n_rows = int(np.ceil(n_videos / n_cols))

    fig = make_subplots(
        rows=n_rows,
        cols=n_cols,
        subplot_titles=list(counts_per_video_frame.keys()),
        horizontal_spacing=0.04,
        vertical_spacing=0.025,
    )

    fps = float(next(iter(dt.leaves)).ds.fps)

    escape_types_in_legend: set[str] = set()

    for i, ky in enumerate(counts_per_video_frame):
        row = i // n_cols + 1
        col = i % n_cols + 1

        video_frame_idcs = np.arange(counts_per_video_frame[ky].shape[0])
        time_min = video_frame_idcs / fps / 60

        fig.add_trace(
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

        selected_frame_idcs = frames_per_video_to_extract[ky]
        show_legend_selected = "selected_frames" not in escape_types_in_legend
        escape_types_in_legend.add("selected_frames")
        fig.add_trace(
            go.Scattergl(
                x=selected_frame_idcs / fps / 60,
                y=counts_per_video_frame[ky][selected_frame_idcs],
                mode="markers",
                marker=dict(symbol="circle", size=2.5, color="orange"),
                name=f"selected frames ({frames_per_video_fraction * 100}%)",
                legendgroup="selected_frames",
                showlegend=show_legend_selected,
                hovertemplate=(
                    "t=%{x:.2f} min<br>n_detections=%{y}<extra></extra>"
                ),
            ),
            row=row,
            col=col,
        )

        escapes_video_frame_idcs = dt[ky].clip_escape_first_frame_0idx.values
        escape_type = dt[ky].clip_escape_type.values

        for x_val, esc_type in zip(
            escapes_video_frame_idcs, escape_type, strict=True
        ):
            color, dash = map_escape_type_to_plotly_style[esc_type]
            x_min = float(x_val) / fps / 60
            show_legend = esc_type not in escape_types_in_legend
            escape_types_in_legend.add(esc_type)
            fig.add_trace(
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

        is_bottom_row = row == n_rows or (
            row == n_rows - 1 and i + n_cols >= n_videos
        )
        fig.update_xaxes(
            title_text="time (min)" if is_bottom_row else None,
            row=row,
            col=col,
        )
        fig.update_yaxes(
            title_text="n detections" if col == 1 else None,
            range=[0, 120],
            row=row,
            col=col,
        )

    for annotation in fig.layout.annotations:
        annotation.font.size = 12

    fig.update_layout(
        height=180 * n_rows,
        width=1500,
        plot_bgcolor="white",
        paper_bgcolor="white",
        legend=dict(x=1.02, y=1, yanchor="top", xanchor="left"),
        margin=dict(l=50, r=50, t=30, b=30),
    )
    fig.update_xaxes(
        showgrid=True,
        gridcolor="lightgrey",
        linecolor="black",
        mirror=True,
        ticks="outside",
        zeroline=False,
    )
    fig.update_yaxes(
        showgrid=True,
        gridcolor="lightgrey",
        linecolor="black",
        mirror=True,
        ticks="outside",
        zeroline=False,
    )

    fig.write_html(str(output_html_path))


def main(args: argparse.Namespace) -> None:
    """Extract low-count frames per video from the input zarr store."""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir_timestamped = Path(f"{args.output_dir}_{timestamp}")
    output_dir_timestamped.mkdir(parents=True, exist_ok=True)

    dt = xr.open_datatree(args.zarr_store, engine="zarr", chunks={})

    counts_per_video_frame: dict[str, np.ndarray] = {}
    frames_per_video_to_extract: dict[str, np.ndarray] = {}
    rows_per_clip: list[dict] = []

    for dt_video in dt.leaves:
        ds_video = dt_video.ds
        video_id = ds_video.video_id

        counts_video = _counts_per_video_frame(ds_video)
        n_to_extract = int(args.frames_per_video_fraction * counts_video.size)

        bottom_idcs = _select_lowest_count_frame_idcs(
            counts_video, n_to_extract
        )

        n_frames_per_clip = (
            ds_video.clip_last_frame_0idx - ds_video.clip_first_frame_0idx + 1
        ).values
        per_clip_idcs = _video_idcs_to_per_clip_idcs(
            bottom_idcs, n_frames_per_clip, ds_video.clip_id.values
        )

        rows_per_clip.extend(
            _build_rows_per_clip(per_clip_idcs, video_id, ds_video)
        )

        counts_per_video_frame[video_id] = counts_video
        frames_per_video_to_extract[video_id] = bottom_idcs

        print(f"{video_id}: {n_to_extract} frames selected")

    csv_path = output_dir_timestamped / "frames_per_clip.csv"
    df_per_clip = pd.DataFrame(rows_per_clip)
    with open(csv_path, "w") as f:
        f.write(
            f"# frames_per_video_fraction={args.frames_per_video_fraction}\n"
        )
        df_per_clip.to_csv(f, index=False)

    if args.save_html_figure:
        plot_n_detections_html(
            counts_per_video_frame,
            frames_per_video_to_extract,
            dt,
            args.frames_per_video_fraction,
            output_dir_timestamped / "n_detections_per_video.html",
        )

    print(f"Output written to {output_dir_timestamped}")


def parse_args(list_args: list[str]) -> argparse.Namespace:
    """Parse CLI args."""
    parser = argparse.ArgumentParser(
        description=(
            "Extract candidate frames for burrow-prompt annotation per video "
            "from a CrabTracks zarr store. The lowest-detection-count frames "
            "are selected, since they correspond to less crowded scenes."
        ),
    )
    parser.add_argument(
        "zarr_store",
        type=Path,
        help=(
            "Path to the input trajectories zarr store. "
            "Usually a CrabTracks zarr file produced by "
            "create-zarr-dataset."
        ),
    )
    parser.add_argument(
        "output_dir",
        type=Path,
        help=(
            "Output directory. A '_<YYYYMMDD_HHMMSS>' suffix is appended to "
            "this path before the directory is created, so multiple runs "
            "with the same output_dir argument never collide. "
            "The combined frames_per_clip.csv (and HTML figure, if "
            "requested) is saved here."
        ),
    )
    parser.add_argument(
        "--frames-per-video-fraction",
        type=float,
        default=0.05,
        help=(
            "Fraction of frames per video to select as lowest-count "
            "candidates (default: 0.05)."
        ),
    )
    parser.add_argument(
        "--save-html-figure",
        action="store_true",
        help=(
            "If set, also save an interactive plotly HTML figure with the "
            "per-video detection counts, selected frames, and escape "
            "markers. Default: not set."
        ),
    )

    return parser.parse_args(list_args)


if __name__ == "__main__":
    args = parse_args(sys.argv[1:])
    main(args)
