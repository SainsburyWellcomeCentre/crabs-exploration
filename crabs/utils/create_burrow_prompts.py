"""Generate SAM3 burrow prompts from crab trajectory zarr stores.

For every video group in a `CrabTracks` zarr store, flatten all trajectories
into 2D points, build a 2D histogram of those points, threshold and smooth it,
detect local maxima as candidate "node" locations, and turn each peak into a
prompt: a point (the peak coordinates, unaltered) and a square bounding box
around it (clipped to the image bounds).

One CSV is written per video into a timestamped output directory. Optionally,
an interactive plotly HTML figure per video is also written, with the
trajectory rasterised by datashader and the prompts overlaid.

Output CSV columns (one row per prompt):
    video_id,
    prompt_point_x, prompt_point_y,
    prompt_bbox_xmin, prompt_bbox_ymin,
    prompt_bbox_xmax, prompt_bbox_ymax,
    prompt_id

Usage (dependencies are auto-installed via uv):
    uv run create_burrow_prompts.py /path/to/store.zarr /path/to/out_dir
    uv run create_burrow_prompts.py /path/to/store.zarr /path/to/out_dir \
        --save-html-figure
"""

# /// script
# requires-python = ">=3.11"
# dependencies = [
#   "numpy>=2.0.0",
#   "pandas",
#   "xarray",
#   "zarr",
#   "scikit-image",
#   "plotly",
#   "datashader",
#   "Pillow",
# ]
# ///

import argparse
import io
import sys
from datetime import datetime
from pathlib import Path

import datashader as ds
import datashader.transfer_functions as tf
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import xarray as xr
from PIL import Image
from skimage.feature import peak_local_max
from skimage.filters import gaussian

# ----------------------------------------------------------------------
# Trajectory & histogram pipeline
# ----------------------------------------------------------------------


def flatten_trajectory_xy(
    ds_video: xr.Dataset,
) -> tuple[np.ndarray, np.ndarray]:
    """Flatten ``position`` over (time, individuals) and drop NaNs."""
    x = ds_video.position.sel(space="x").values.reshape(-1)
    y = ds_video.position.sel(space="y").values.reshape(-1)
    mask = ~np.isnan(x) & ~np.isnan(y)
    return x[mask], y[mask]


def compute_2d_histogram(
    points_x: np.ndarray,
    points_y: np.ndarray,
    image_w: int,
    image_h: int,
    bin_size_pixels: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Bin trajectory points into a 2D histogram over the image."""
    n_bins_x = round(image_w / bin_size_pixels)
    n_bins_y = round(image_h / bin_size_pixels)
    counts, xedges, yedges = np.histogram2d(
        points_x,
        points_y,
        bins=[n_bins_x, n_bins_y],
        range=[[0, image_w], [0, image_h]],
    )
    return counts, xedges, yedges


def threshold_counts(counts: np.ndarray, percentile: float) -> np.ndarray:
    """Zero histogram bins whose counts are below the given percentile."""
    min_count = np.percentile(counts, percentile)
    return np.where(counts >= min_count, counts, 0)


def log_transform_counts(counts_filtered: np.ndarray) -> np.ndarray:
    """Apply ``log1p`` to the filtered counts."""
    return np.log1p(counts_filtered)


def smooth_counts(log_counts: np.ndarray, sigma: float) -> np.ndarray:
    """Apply a Gaussian filter to the log-transformed counts."""
    return gaussian(log_counts, sigma=sigma, preserve_range=True)


def find_peak_bin_indices(
    smoothed: np.ndarray,
    min_distance: int,
    threshold_rel: float,
) -> np.ndarray:
    """Detect local maxima in the smoothed histogram, in bin-index space."""
    return peak_local_max(
        smoothed,
        min_distance=min_distance,
        threshold_rel=threshold_rel,
    )


def bin_indices_to_pixels(
    peaks_col_row: np.ndarray,
    xedges: np.ndarray,
    yedges: np.ndarray,
) -> np.ndarray:
    """Convert (col, row) bin indices to (x, y) pixel coordinates."""
    x_bin_centers = (xedges[:-1] + xedges[1:]) / 2
    y_bin_centers = (yedges[:-1] + yedges[1:]) / 2
    node_x = x_bin_centers[peaks_col_row[:, 0]]
    node_y = y_bin_centers[peaks_col_row[:, 1]]
    return np.column_stack([node_x, node_y])


# ----------------------------------------------------------------------
# Point prompts -> bbox prompts
# ----------------------------------------------------------------------


def point_prompts_to_bbox_prompts(
    peaks_xy: np.ndarray,
    node_radius_pixels: float,
    image_w: int,
    image_h: int,
) -> np.ndarray:
    """Build square bboxes around each peak and clip them to the image."""
    node_x = peaks_xy[:, 0]
    node_y = peaks_xy[:, 1]
    return np.column_stack(
        [
            np.clip(node_x - node_radius_pixels, 0, image_w),
            np.clip(node_y - node_radius_pixels, 0, image_h),
            np.clip(node_x + node_radius_pixels, 0, image_w),
            np.clip(node_y + node_radius_pixels, 0, image_h),
        ]
    )


# ----------------------------------------------------------------------
# CSV assembly
# ----------------------------------------------------------------------


def prompts_to_dataframe(
    peaks_xy: np.ndarray,
    bboxes_clipped_x1y1x2y2: np.ndarray,
    video_id: str,
) -> pd.DataFrame:
    """Build the per-video prompts dataframe."""
    n = peaks_xy.shape[0]
    return pd.DataFrame(
        {
            "video_id": [video_id] * n,
            "prompt_point_x": peaks_xy[:, 0],
            "prompt_point_y": peaks_xy[:, 1],
            "prompt_bbox_xmin": bboxes_clipped_x1y1x2y2[:, 0],
            "prompt_bbox_ymin": bboxes_clipped_x1y1x2y2[:, 1],
            "prompt_bbox_xmax": bboxes_clipped_x1y1x2y2[:, 2],
            "prompt_bbox_ymax": bboxes_clipped_x1y1x2y2[:, 3],
            "prompt_id": np.arange(n, dtype=int),
        }
    )


# ----------------------------------------------------------------------
# Plotly figure
# ----------------------------------------------------------------------


def rasterise_trajectory(
    points_x: np.ndarray,
    points_y: np.ndarray,
    image_w: int,
    image_h: int,
    dynspread_threshold: float = 0.975,
) -> Image.Image:
    """Rasterise trajectory points with datashader to a PIL image."""
    canvas = ds.Canvas(
        plot_width=image_w,
        plot_height=image_h,
        x_range=(0, image_w),
        y_range=(0, image_h),
    )
    agg = canvas.points(pd.DataFrame({"x": points_x, "y": points_y}), "x", "y")
    img = tf.shade(agg, cmap=["#1f77b4"])
    img = tf.dynspread(img, threshold=dynspread_threshold)
    return img.to_pil().transpose(Image.FLIP_TOP_BOTTOM)


def build_trajectory_figure(
    rasterised_pil_image: Image.Image,
    video_id: str,
    video_length_minutes: float,
    image_w: int,
    image_h: int,
) -> go.Figure:
    """Wrap the rasterised trajectory image in a plotly figure."""
    img_buffer = io.BytesIO()
    rasterised_pil_image.save(img_buffer, format="PNG")
    img_buffer.seek(0)

    fig = go.Figure()
    fig.add_layout_image(
        source=Image.open(img_buffer),
        xref="x",
        yref="y",
        x=0,
        y=0,
        sizex=image_w,
        sizey=image_h,
        sizing="stretch",
        layer="below",
    )
    fig.update_layout(
        title=f"{video_id} ({video_length_minutes:.1f} min)",
        xaxis_title="x (pixels)",
        yaxis_title="y (pixels)",
        yaxis_scaleanchor="x",
        plot_bgcolor="white",
        paper_bgcolor="white",
        xaxis=dict(
            showgrid=True,
            gridcolor="lightgrey",
            gridwidth=0.5,
            zeroline=False,
            linecolor="black",
            mirror=True,
            ticks="outside",
            range=[0, image_w],
        ),
        yaxis=dict(
            showgrid=True,
            gridcolor="lightgrey",
            gridwidth=0.5,
            zeroline=False,
            linecolor="black",
            mirror=True,
            ticks="outside",
            range=[image_h, 0],
        ),
    )
    return fig


def overlay_prompts(
    fig: go.Figure,
    peaks_xy: np.ndarray,
    bboxes_clipped_x1y1x2y2: np.ndarray,
) -> None:
    """Add red bbox rectangles and red 'x' peak markers to the figure."""
    n = peaks_xy.shape[0]

    fig.add_trace(
        go.Scattergl(
            x=peaks_xy[:, 0],
            y=peaks_xy[:, 1],
            mode="markers",
            marker=dict(symbol="x", color="red", size=8),
            name=f"{n} peaks",
        )
    )

    for x1, y1, x2, y2 in bboxes_clipped_x1y1x2y2:
        fig.add_shape(
            type="rect",
            x0=x1,
            y0=y1,
            x1=x2,
            y1=y2,
            line=dict(color="red", width=1),
        )

    current_title = fig.layout.title.text or ""
    fig.update_layout(title=f"{current_title} \u2014 {n} prompts")


def plot_prompts_html(
    points_x: np.ndarray,
    points_y: np.ndarray,
    peaks_xy: np.ndarray,
    bboxes_clipped_x1y1x2y2: np.ndarray,
    video_id: str,
    video_length_minutes: float,
    image_w: int,
    image_h: int,
    output_html_path: Path,
) -> None:
    """Build and save the per-video prompts overlay figure as HTML."""
    rasterised = rasterise_trajectory(points_x, points_y, image_w, image_h)
    fig = build_trajectory_figure(
        rasterised, video_id, video_length_minutes, image_w, image_h
    )
    overlay_prompts(fig, peaks_xy, bboxes_clipped_x1y1x2y2)
    fig.write_html(str(output_html_path))


# ----------------------------------------------------------------------
# Per-video orchestration
# ----------------------------------------------------------------------


def get_video_length_minutes(ds_video: xr.Dataset) -> float:
    """Return the video length in minutes from the zarr coords/attrs."""
    n_frames = int(ds_video.clip_last_frame_0idx.max()) + 1
    return n_frames / float(ds_video.fps) / 60


def process_video(
    ds_video: xr.Dataset,
    video_id: str,
    output_dir: Path,
    *,
    save_html: bool,
    image_w: int,
    image_h: int,
    bin_size_pixels: int,
    counts_percentile: float,
    gaussian_sigma: float,
    peaks_min_distance: int,
    peaks_threshold_rel: float,
    node_radius_pixels: float,
) -> int:
    """Run the full pipeline for one video and write its outputs."""
    points_x, points_y = flatten_trajectory_xy(ds_video)

    counts, xedges, yedges = compute_2d_histogram(
        points_x, points_y, image_w, image_h, bin_size_pixels
    )
    counts_filtered = threshold_counts(counts, counts_percentile)
    log_counts = log_transform_counts(counts_filtered)
    smoothed = smooth_counts(log_counts, gaussian_sigma)
    peaks_col_row = find_peak_bin_indices(
        smoothed, peaks_min_distance, peaks_threshold_rel
    )
    peaks_xy = bin_indices_to_pixels(peaks_col_row, xedges, yedges)
    bboxes_clipped = point_prompts_to_bbox_prompts(
        peaks_xy, node_radius_pixels, image_w, image_h
    )

    df = prompts_to_dataframe(peaks_xy, bboxes_clipped, video_id)
    df.to_csv(output_dir / f"{video_id}.csv", index=False)

    if save_html:
        plot_prompts_html(
            points_x,
            points_y,
            peaks_xy,
            bboxes_clipped,
            video_id,
            get_video_length_minutes(ds_video),
            image_w,
            image_h,
            output_dir / f"{video_id}.html",
        )

    return peaks_xy.shape[0]


# ----------------------------------------------------------------------
# Top-level
# ----------------------------------------------------------------------


def main(args: argparse.Namespace) -> None:
    """Generate burrow prompts for every video in the zarr store."""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    actual_output_dir = Path(f"{args.output_dir}_{timestamp}")
    actual_output_dir.mkdir(parents=True, exist_ok=True)

    if args.node_radius_pixels is None:
        node_radius_pixels = args.gaussian_sigma * 4 * args.bin_size_pixels
    else:
        node_radius_pixels = args.node_radius_pixels

    dt = xr.open_datatree(args.zarr_store, engine="zarr", chunks={})

    for video_id, video_node in dt.children.items():
        ds_video = video_node.ds
        n_prompts = process_video(
            ds_video,
            video_id,
            actual_output_dir,
            save_html=args.save_html_figure,
            image_w=args.image_width,
            image_h=args.image_height,
            bin_size_pixels=args.bin_size_pixels,
            counts_percentile=args.counts_percentile,
            gaussian_sigma=args.gaussian_sigma,
            peaks_min_distance=args.peaks_min_distance,
            peaks_threshold_rel=args.peaks_threshold_rel,
            node_radius_pixels=node_radius_pixels,
        )
        print(f"Video {video_id}: {n_prompts} prompts")

    print(f"Output written to {actual_output_dir}")


def parse_args(list_args: list[str]) -> argparse.Namespace:
    """Parse CLI args."""
    parser = argparse.ArgumentParser(
        description=(
            "Generate SAM3 burrow point and bbox prompts from a CrabTracks "
            "zarr store, one CSV per video."
        ),
    )
    parser.add_argument(
        "zarr_store",
        type=Path,
        help=(
            "Path to the CrabTracks zarr store produced by "
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
            "Per-video CSVs (and HTMLs, if requested) are written inside."
        ),
    )
    parser.add_argument(
        "--bin-size-pixels",
        type=int,
        default=5,
        help=(
            "Edge length in pixels of one bin in the trajectory 2D "
            "histogram (default: 5)."
        ),
    )
    parser.add_argument(
        "--counts-percentile",
        type=float,
        default=99,
        help=(
            "Histogram bins with counts below this percentile are zeroed "
            "before smoothing (default: 99)."
        ),
    )
    parser.add_argument(
        "--gaussian-sigma",
        type=float,
        default=2.5,
        help=(
            "Standard deviation in bins of the Gaussian filter applied to "
            "the log-transformed counts (default: 2.5)."
        ),
    )
    parser.add_argument(
        "--peaks-min-distance",
        type=int,
        default=10,
        help=(
            "Minimum separation in bins between two detected peaks, passed "
            "to skimage.feature.peak_local_max (default: 10)."
        ),
    )
    parser.add_argument(
        "--peaks-threshold-rel",
        type=float,
        default=0.5,
        help=(
            "Minimum peak intensity relative to the maximum, passed to "
            "skimage.feature.peak_local_max (default: 0.5)."
        ),
    )
    parser.add_argument(
        "--node-radius-pixels",
        type=float,
        default=None,
        help=(
            "Half side length in pixels of the square bbox drawn around "
            "each peak before clipping. Defaults to "
            "gaussian_sigma * 4 * bin_size_pixels."
        ),
    )
    parser.add_argument(
        "--image-width",
        type=int,
        default=4096,
        help=(
            "Image width in pixels; used as the upper x-range of the "
            "histogram and to clip bboxes (default: 4096). The zarr store "
            "does not currently expose image dimensions in its attrs."
        ),
    )
    parser.add_argument(
        "--image-height",
        type=int,
        default=2160,
        help=(
            "Image height in pixels; used as the upper y-range of the "
            "histogram and to clip bboxes (default: 2160)."
        ),
    )
    parser.add_argument(
        "--save-html-figure",
        action="store_true",
        help=(
            "If set, also save an interactive plotly HTML figure per video, "
            "with the trajectory rasterised by datashader and the detected "
            "peaks and prompt bboxes overlaid. Default: not set."
        ),
    )
    return parser.parse_args(list_args)


if __name__ == "__main__":
    main(parse_args(sys.argv[1:]))
