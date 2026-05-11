"""Compute frames to extract for prompting SAM3 to detect burrows.

For every video in the input zarr store, we compute the number of crab
detections per frame, and select a fixed fraction of frames with the lowest
detection counts.

The output consists of:
* a CSV file with 0-based frame indices, defined per-clip and per-video, and
* optionally, a plotly HTML figure with the detection counts per frame,
  per video and the selected frames.
Outputs are saved to a user defined location, to which a timestamp is appended.

The structure of the CSV is as follows:
* Columns (one row per selected frame):
    loop_clip_name,
    frame_0idx_in_clip,
    video_name,
    frame_0idx_in_video.

* A leading ``# frames_per_video_fraction=...`` comment row is written to
document the fraction used (it can be skipped with
``pd.read_csv(path, comment="#")``).

Usage (dependencies are auto-installed via uv):
* Default
    uv run compute_frames_for_burrow_prompts.py /path/to/store.zarr
    /path/to/out_dir
* Non-default fraction
    uv run compute_frames_for_burrow_prompts.py /path/to/store.zarr
    /path/to/out_dir --frames-per-video-fraction 0.1
* Save HTML figure
    uv run compute_frames_for_burrow_prompts.py /path/to/store.zarr
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
    """Compute detections per video frame in input video dataset.

    Concatenates per-clip non-null ``confidence`` counts (truncated to each
    clip's actual frame count) into a single array of length N, where N is
    the total number of frames in the video.
    """
    counts_per_clip_in_video = (
        (~ds_video.confidence.isnull()).sum(axis=-1).compute().values
    )  # (clips, max_frames_per_clip)

    n_frames_per_clip = (
        ds_video.clip_last_frame_0idx - ds_video.clip_first_frame_0idx + 1
    ).values  # (clips)

    counts_per_video_frame = np.concatenate(
        [
            counts_per_clip_in_video[i, :n]
            for i, n in enumerate(n_frames_per_clip)
        ]
    )  # (N,), where N is total video frames

    return counts_per_video_frame


def _select_lowest_count_frame_idcs(
    counts_video: np.ndarray,
    frames_fraction: float,
) -> np.ndarray:
    """Return indices of the ``n_to_extract`` frames with the lowest counts.

    Indices are sorted by count (ascending). Ties are broken by earliest
    frame index (stable sort).
    """
    n_to_extract = int(frames_fraction * counts_video.size)
    bottom_idcs = np.argpartition(counts_video, n_to_extract)[:n_to_extract]
    return bottom_idcs[np.argsort(counts_video[bottom_idcs], kind="stable")]


def _video_idcs_to_per_clip_idcs(
    video_based_frame_idcs: np.ndarray, ds_video: xr.Dataset
) -> dict[str, np.ndarray]:
    """Convert video-based frame indices to clip-based frame indices.

    Returns a dict mapping ``clip_id`` to the array of frame indices within
    that clip (preserving the input order). Clips with no selected frames are
    omitted.
    """
    # Compute n frames per clip
    n_frames_per_clip = (
        ds_video.clip_last_frame_0idx - ds_video.clip_first_frame_0idx + 1
    ).values

    # Compute start and end frame (video-based) per clip
    clip_boundaries = np.concatenate([[0], np.cumsum(n_frames_per_clip)])

    # Get the clip id (as integer) each video frame is in
    clip_id_per_frame = (
        np.searchsorted(clip_boundaries, video_based_frame_idcs, side="right")
        - 1
    )

    # Compute per clip frame idcs
    clip_based_frame_idcs = (
        video_based_frame_idcs - clip_boundaries[clip_id_per_frame]
    )

    # Split results per clip
    frame_idcs_per_clip: dict[str, np.ndarray] = {}
    for i, clip_id in enumerate(ds_video.clip_id.values):
        # get only frame idcs for clip "i"
        mask = clip_id_per_frame == i
        if mask.any():
            frame_idcs_per_clip[clip_id] = clip_based_frame_idcs[mask]
    return frame_idcs_per_clip


def _build_rows_per_clip(
    frame_idcs_per_clip: dict[str, np.ndarray],
    frame_idcs_video_based: np.ndarray,
    ds_video: xr.Dataset,
) -> list[dict]:
    """Build CSV rows for one video from the per-clip selected indices.

    Note that frame_idcs_video_based is sorted by increasing detection count,
    (which does not match the order of frame indices in `frame_idcs_per_clip`
    if we visit them per clip). It is used in this function as a sanity
    check only.
    """
    rows = []
    video_id = ds_video.video_id
    # Loop thru clips in video and append each frame to extract
    for clip_id, frame_idcs_clip_based in frame_idcs_per_clip.items():
        clip_start = ds_video.clip_first_frame_0idx.sel(clip_id=clip_id).item()
        for f in frame_idcs_clip_based:
            # Check video-based frame to log is in `frame_idcs_video_based`
            assert int(f) + clip_start in frame_idcs_video_based

            # append to list of csv rows
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
    """Plot per-video detection counts with selected frames.

    The figure includes one subplot per video, showing detections per frame,
    markers at the selected (lowest-count) frames, and vertical lines at escape
    frames coloured by escape type.
    """
    # Define line style per escape type
    # (the vertical line is shown at the start of the escape)
    map_escape_type_to_plotly_style = {
        "triggered": ("red", "solid"),
        "spontaneous": ("red", "dash"),
        "tourists": ("green", "solid"),
    }

    # Get fps
    fps = float(next(iter(dt.leaves)).ds.fps)

    # Set max range y axis
    y_max_plot = 120

    # Define grid of plots
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

    # Loop thru video
    legend_set: set[str] = set()
    for i, video_id in enumerate(counts_per_video_frame):
        # Get position in grid
        # (i % n_cols cycles 0, 1, 2, 0, 1, 2, ...)
        row = i // n_cols + 1
        col = i % n_cols + 1

        # Compute x-axis values (later expressed in minutes)
        all_video_frame_idcs = np.arange(
            counts_per_video_frame[video_id].shape[0]
        )

        # Plot detection count per frame
        fig.add_trace(
            go.Scattergl(
                x=all_video_frame_idcs / fps / 60,
                y=counts_per_video_frame[video_id],
                mode="lines",
                line=dict(color="#1f77b4", width=1),
                name=video_id,
                showlegend=False,
                hovertemplate=(
                    "t=%{x:.2f} min<br>n_detections=%{y}<extra></extra>"
                ),  # show time in min and detections on hover;
                # <extra></extra> hides the default box Plotly
                # normally appends
            ),
            row=row,
            col=col,
        )

        # Mark selected frames to extract
        selected_frame_idcs = frames_per_video_to_extract[video_id]
        # prevent duplicate legend entries
        show_legend_selected = "selected_frames" not in legend_set
        legend_set.add("selected_frames")
        fig.add_trace(
            go.Scattergl(
                x=selected_frame_idcs / fps / 60,
                y=counts_per_video_frame[video_id][selected_frame_idcs],
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

        # Plot vertical lines to mark escapes
        escape_start_frame_idcs = dt[
            video_id
        ].clip_escape_first_frame_0idx.values
        escape_types = dt[video_id].clip_escape_type.values

        for x_val, esc_type in zip(
            escape_start_frame_idcs, escape_types, strict=True
        ):
            # Get x coordinate for this escape
            x_min = float(x_val) / fps / 60

            # Get line style
            color, dash = map_escape_type_to_plotly_style[esc_type]

            # avoid duplicates in legend
            show_legend = esc_type not in legend_set
            legend_set.add(esc_type)
            fig.add_trace(
                go.Scattergl(
                    x=[x_min, x_min],
                    y=[0, y_max_plot],
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

        # Label x axis only for bottom row
        is_bottom_row = row == n_rows or (
            row == n_rows - 1 and i + n_cols >= n_videos
        )
        fig.update_xaxes(
            title_text="time (min)" if is_bottom_row else None,
            row=row,
            col=col,
        )
        # Label y axis only for left-most column
        fig.update_yaxes(
            title_text="n detections" if col == 1 else None,
            range=[0, y_max_plot],
            row=row,
            col=col,
        )

    # Set fontsize for plot titles
    for annotation in fig.layout.annotations:
        annotation.font.size = 12

    # Change fig params to matplotlib-like style
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
    # Create output directory
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir_timestamped = Path(f"{args.output_dir}_{timestamp}")
    output_dir_timestamped.mkdir(parents=True, exist_ok=True)

    # Read zarr store as datatree
    dt = xr.open_datatree(args.zarr_store, engine="zarr", chunks={})

    # Loop thru videos
    counts_per_video_frame: dict[str, np.ndarray] = {}
    frames_per_video_to_extract: dict[str, np.ndarray] = {}
    list_csv_rows: list[dict] = []
    for dt_video in dt.leaves:
        # Get video dataset and id
        ds_video = dt_video.ds
        video_id = ds_video.video_id

        # Compute array with counts per video frame
        counts_video = _counts_per_video_frame(ds_video)

        # Compute indices of X% of frames with lowest count
        # (returns an array of size N, N = n of frames in video,
        # sorted by count)
        frame_idcs_video_based = _select_lowest_count_frame_idcs(
            counts_video,
            args.frames_per_video_fraction,
        )

        # Transform video-based indices to clip-based indices
        # (returns a dict mapping clip_id (str) to clip-based indices,
        # sorted by count within clip)
        frame_idcs_clip_based = _video_idcs_to_per_clip_idcs(
            frame_idcs_video_based, ds_video
        )

        # Extend csv rows list with new data
        # Note: frame_idcs_video_based just used for check
        list_csv_rows.extend(
            _build_rows_per_clip(
                frame_idcs_clip_based,
                frame_idcs_video_based,
                ds_video,
            )
        )

        # If exporting figure, save required data
        if args.save_html_figure:
            counts_per_video_frame[video_id] = counts_video
            frames_per_video_to_extract[video_id] = frame_idcs_video_based

    # Save csv
    csv_path = output_dir_timestamped / "frames_to_extract.csv"
    df_per_clip = pd.DataFrame(list_csv_rows)
    with open(csv_path, "w") as f:
        f.write(
            f"# frames_per_video_fraction={args.frames_per_video_fraction}\n"
        )
        df_per_clip.to_csv(f, index=False)

    # Plot html figure
    if args.save_html_figure:
        plot_n_detections_html(
            counts_per_video_frame,
            frames_per_video_to_extract,
            dt,
            args.frames_per_video_fraction,
            output_dir_timestamped / "frames_to_extract.html",
        )

    print(f"Output written to {output_dir_timestamped}")


def parse_args(list_args: list[str]) -> argparse.Namespace:
    """Parse CLI args."""
    parser = argparse.ArgumentParser(
        description=(
            "Compute frames to extract for prompting SAM3 to detect burrows. "
            "The lowest-detection-count frames "
            "are selected, since they correspond to less crowded scenes."
        ),
    )
    parser.add_argument(
        "zarr_store",
        type=Path,
        help=(
            "Path to the zarr store with the trajectory data. "
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
            "The frames_to_extract.csv (and HTML figure, if "
            "requested) is saved here."
        ),
    )
    parser.add_argument(
        "--frames-per-video-fraction",
        type=float,
        default=0.05,
        help=(
            "Fraction of frames per video to extract (default: 0.05). "
            "Before extracting,"
            "frames are sorted in ascending count order."
        ),
    )
    parser.add_argument(
        "--save-html-figure",
        action="store_true",
        help=(
            "If passed, a plotly HTML figure is saved with the "
            "per-video detection counts, selected frames, and escape "
            "markers. Default: not set."
        ),
    )

    return parser.parse_args(list_args)


if __name__ == "__main__":
    args = parse_args(sys.argv[1:])
    main(args)
