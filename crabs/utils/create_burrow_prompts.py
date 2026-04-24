"""Generate prompts to segment burrows with SAM3 from crab trajectory data.

For every video group in the input zarr store, we collect trajectories across
all clips, discretise them into a 2D histogram, remove bins below a threshold
and apply Gaussian smoothing to the result. We then detect local maxima as
candidate burrow locations (trajectory "nodes"), and turn each of these into
prompts of two formats: a point one (the peak coordinates, unaltered) and a
square bounding box around it (clipped to the image bounds).

The results are written into a CSV per video, under a timestamped output
directory. Optionally, an interactive plotly HTML figure per video is also
saved, with the rasterised trajectory data and the prompts overlaid.

Output CSV columns (one row per prompt):
    group_id,
    prompt_point_x,prompt_point_y,
    prompt_bbox_xmin,prompt_bbox_ymin,
    prompt_bbox_xmax,prompt_bbox_ymax,
    prompt_id, # id within group
    peak_value_rel # peak value relative to max

Usage (dependencies are auto-installed via uv):
* To generate prompts per video (default)
    uv run create_burrow_prompts.py /path/to/store.zarr /path/to/out_dir
* To generate prompts per day
    uv run create_burrow_prompts.py /path/to/store.zarr /path/to/out_dir \
        --group-by-pattern
* To generate prompts per specific pattern (e.g. across all Sept data)
    uv run create_burrow_prompts.py /path/to/store.zarr /path/to/out_dir \
        --group-by-pattern *.09.*
* To save figures
    uv run create_burrow_prompts.py /path/to/store.zarr /path/to/out_dir \
        --save-html-figure


"""

# /// script
# requires-python = ">=3.11"
# dependencies = [
#   "numpy>=2.0.0",
#   "pandas",
#   "xarray",
#   "dask",
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
import plotly.colors as pc
import plotly.graph_objects as go
import xarray as xr
from PIL import Image
from skimage.feature import peak_local_max
from skimage.filters import gaussian


def prompts_from_trajectory_data(
    trajectories_xy: np.ndarray,
    *,
    image_w: int,
    image_h: int,
    bin_size_pixels: int,
    counts_percentile: float,
    gaussian_sigma: float,
    peaks_min_distance: int,
    peaks_threshold_rel: float,
    node_radius_pixels: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Compute SAM3 prompts for one group of trajectories.

    A group can be a single video or the union of several (e.g. all videos
    from a given date).
    """
    # Compute 2D histogram (square bins)
    counts, xedges, yedges = _compute_2d_histogram(
        trajectories_xy, image_w, image_h, bin_size_pixels
    )

    # Postprocess
    smoothed = _apply_threshold_log_gaussian(
        counts, counts_percentile, gaussian_sigma
    )

    # Detect local maxima in the smoothed histogram, in bin-index space
    peaks_col_row = peak_local_max(
        smoothed,
        min_distance=peaks_min_distance,
        threshold_rel=peaks_threshold_rel,
    )
    peak_values = smoothed[peaks_col_row[:, 0], peaks_col_row[:, 1]]
    peak_values_rel = peak_values / smoothed.max()

    # Transform peaks coords in bin-index space into prompts
    # Set default radius for node around peak if required
    if node_radius_pixels is None:
        node_radius_pixels = gaussian_sigma * 4 * bin_size_pixels
    peaks_xy, bboxes_clipped_x1y1x2y2 = _peaks_to_prompts(
        peaks_col_row,
        xedges,
        yedges,
        node_radius_pixels,
        image_w,
        image_h,
    )

    return peaks_xy, bboxes_clipped_x1y1x2y2, peak_values_rel


def _flatten_trajectory_xy(
    ds_video: xr.Dataset,
) -> np.ndarray:
    """Flatten ``position`` over (time, individuals) and drop NaNs.

    We construct a 2D array of coordinates aggregating all individuals.
    """
    x = ds_video.position.sel(space="x").values.reshape(-1)
    y = ds_video.position.sel(space="y").values.reshape(-1)
    mask = ~np.isnan(x) & ~np.isnan(y)
    return np.c_[x[mask], y[mask]]


def _compute_2d_histogram(
    points_xy: np.ndarray,
    image_w: int,
    image_h: int,
    bin_size_pixels: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Bin trajectory points into a 2D histogram over the image.

    Bins are square.
    """
    n_bins_x = round(image_w / bin_size_pixels)
    n_bins_y = round(image_h / bin_size_pixels)
    counts, xedges, yedges = np.histogram2d(
        points_xy[:, 0],
        points_xy[:, 1],
        bins=[n_bins_x, n_bins_y],
        range=[[0, image_w], [0, image_h]],
    )
    return counts, xedges, yedges


def _apply_threshold_log_gaussian(counts, percentile, sigma):
    """Postprocess the histogram counts.

    The steps are:
    - Set to zero histogram bins whose counts are below the given percentile.
    - Take log(x+1) (avoids large bins dominanting)
    - Apply Gaussian filter to histogram
    """
    # Set to zero histogram bins whose counts are below the given percentile.
    min_count = np.percentile(counts, percentile)
    counts_filtered = np.where(counts >= min_count, counts, 0)

    # Take log(x+1)
    log_counts = np.log1p(counts_filtered)

    # Gaussian filter to histogram
    return gaussian(log_counts, sigma=sigma, preserve_range=True)


def _peaks_to_prompts(
    peaks_col_row: np.ndarray,
    xedges: np.ndarray,
    yedges: np.ndarray,
    node_radius_pixels: float,
    image_w: int,
    image_h: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Compute point and bbox prompts from peak values.

    Bboxes are expressed as xmin, ymin, xmax, ymax coordinates.
    """
    # Compute point prompts in pixel space from peak bins
    peaks_xy = _bin_indices_to_pixels(peaks_col_row, xedges, yedges)

    # Compute bbox prompts from peaks in pixel space, clipped
    # to image boundaries
    bboxes_clipped = _point_prompts_to_bbox_prompts(
        peaks_xy, node_radius_pixels, image_w, image_h
    )
    return peaks_xy, bboxes_clipped


def _bin_indices_to_pixels(
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


def _point_prompts_to_bbox_prompts(
    peaks_xy: np.ndarray,
    node_radius_pixels: float,
    image_w: int,
    image_h: int,
) -> np.ndarray:
    """Build square bboxes around each peak and clip to the image boundaries.

    Bboxes are expressed as xmin, ymin, xmax, ymax coordinates.
    """
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


def _get_video_length_minutes(ds_video: xr.Dataset) -> float:
    """Return the video length in minutes from the zarr coords/attrs.

    A clip goes from end of previous escape (or start of video if there is
    no previous escape) to end of current escape.
    """
    n_frames = int(ds_video.clip_last_frame_0idx.max().compute()) + 1
    return n_frames / float(ds_video.fps) / 60


def plot_prompts_html(
    trajectory_points_xy: np.ndarray,
    peaks_xy: np.ndarray,
    peak_values_rel: np.ndarray,
    bboxes_clipped_x1y1x2y2: np.ndarray,
    video_id: str,
    video_length_mins: float,
    image_w: int,
    image_h: int,
    peaks_threshold_rel: float,
    output_html_path: Path,
    colorscale: str = "Plasma",
    dynspread_threshold: float = 0.975,
) -> None:
    """Plot trajectories in single video and overlay per-video prompts.

    Rasterises the trajectory data with datashader and overlays prompts as
    'x' peak markers and bbox rectangles, both coloured by relative peak
    intensity. The figure is saved as an html file.
    """
    # Initialise canvas
    canvas = ds.Canvas(
        plot_width=image_w,
        plot_height=image_h,
        x_range=(0, image_w),
        y_range=(0, image_h),
    )

    # Add points to the canvas and rasterise
    agg = canvas.points(
        pd.DataFrame(
            {
                "x": trajectory_points_xy[:, 0],
                "y": trajectory_points_xy[:, 1],
            }
        ),
        "x",
        "y",
    )
    img = tf.shade(agg, cmap=["#1f77b4"])
    img = tf.dynspread(img, threshold=dynspread_threshold)

    # Convert img to PIL and then to bytes for plotly
    rasterised_pil_image = img.to_pil().transpose(Image.FLIP_TOP_BOTTOM)
    img_buffer = io.BytesIO()
    rasterised_pil_image.save(img_buffer, format="PNG")
    img_buffer.seek(0)

    fig = go.Figure()

    # plot rasterised trajectory data
    fig.add_layout_image(
        source=Image.open(img_buffer),
        xref="x",
        yref="y",
        x=0,
        y=0,  # top-left corner in data coords (y is inverted)
        sizex=image_w,
        sizey=image_h,
        sizing="stretch",
        layer="below",
    )

    # customise axes and labels
    n = peaks_xy.shape[0]
    fig.update_layout(
        title=(f"{video_id} ({video_length_mins:.1f} min) - {n} prompts"),
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
        legend=dict(x=1.02, y=1, yanchor="top", xanchor="left"),
    )

    # Color scale bounds from the peak values (shared by points and bboxes)
    cmin = peaks_threshold_rel
    cmax = 1.0
    norm_values = (peak_values_rel - cmin) / (cmax - cmin)
    bbox_colors = pc.sample_colorscale(colorscale, norm_values)

    # Overlay point prompts as 'x' markers coloured by relative peak intensity
    fig.add_trace(
        go.Scattergl(
            x=peaks_xy[:, 0],
            y=peaks_xy[:, 1],
            mode="markers",
            marker=dict(
                symbol="x",
                size=8,
                color=peak_values_rel,
                colorscale=colorscale,
                cmin=cmin,
                cmax=cmax,
                colorbar=dict(
                    title="peak_value_rel",
                    x=1.02,
                    y=0,
                    yanchor="bottom",
                    len=0.6,
                    thickness=15,
                ),
            ),
            name="prompt_point",
            showlegend=True,
            customdata=peak_values_rel,
            hovertemplate=(
                "x=%{x:.1f}<br>y=%{y:.1f}<br>peak_value_rel=%{customdata:.3f}"
            ),
        )
    )
    # Overlay bbox prompts as rectangles coloured by relative peak intensity
    # (one trace per bbox, but grouped under a single legend entry)
    for i, (xmin, ymin, xmax, ymax) in enumerate(bboxes_clipped_x1y1x2y2):
        fig.add_trace(
            go.Scattergl(
                x=[xmin, xmax, xmax, xmin, xmin],
                y=[ymin, ymin, ymax, ymax, ymin],
                mode="lines",
                line=dict(color=bbox_colors[i], width=1),
                name="prompt_box",
                legendgroup="prompt_box",
                showlegend=(i == 0),
                hoverinfo="skip",
            )
        )

    # Export as html
    fig.write_html(str(output_html_path))


def main(args: argparse.Namespace) -> None:
    """Generate burrow prompts for every video in the zarr store."""
    # Compute timestamped output dir
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir_timestamped = Path(f"{args.output_dir}_{timestamp}")
    output_dir_timestamped.mkdir(parents=True, exist_ok=True)

    # Read zarr store as datatree
    dt = xr.open_datatree(args.zarr_store, engine="zarr", chunks={})

    # ---------------
    # Build the list of (group_id, [leaves]) groups to process.
    # With --group-by-pattern: one group per pattern in --patterns (union
    # across matching videos).
    # Otherwise: one group per video.
    if args.group_by_pattern:
        list_groups = []
        for pattern in args.patterns:
            dt_matched = dt.match(pattern)
            if not dt_matched.children:
                print(f"No videos match {pattern}")
                continue
            leaves = list(dt_matched.leaves)
            group_id = pattern.rstrip("*")
            list_groups.append((group_id, leaves))
    # Otherwise: one group per video
    else:
        list_groups = [(node.ds.video_id, [node]) for node in dt.leaves]
    # ---------------

    # Process each group of trajectories
    for group_id, leaves in list_groups:
        trajectories_xy = np.concatenate(
            [_flatten_trajectory_xy(node.ds) for node in leaves],
            axis=0,
        )
        video_length_minutes = sum(
            _get_video_length_minutes(node.ds) for node in leaves
        )

        peaks_xy, bboxes_clipped_x1y1x2y2, peak_values_rel = (
            prompts_from_trajectory_data(
                trajectories_xy,
                image_w=args.image_width,
                image_h=args.image_height,
                bin_size_pixels=args.bin_size_pixels,
                counts_percentile=args.counts_percentile,
                gaussian_sigma=args.gaussian_sigma,
                peaks_min_distance=args.peaks_min_distance,
                peaks_threshold_rel=args.peaks_threshold_rel,
                node_radius_pixels=args.node_radius_pixels,
            )
        )

        # Export as csv
        n_peaks = peaks_xy.shape[0]
        df = pd.DataFrame(
            {
                "group_id": [group_id] * n_peaks,
                "prompt_point_x": peaks_xy[:, 0],
                "prompt_point_y": peaks_xy[:, 1],
                "prompt_bbox_xmin": bboxes_clipped_x1y1x2y2[:, 0],
                "prompt_bbox_ymin": bboxes_clipped_x1y1x2y2[:, 1],
                "prompt_bbox_xmax": bboxes_clipped_x1y1x2y2[:, 2],
                "prompt_bbox_ymax": bboxes_clipped_x1y1x2y2[:, 3],
                "prompt_id": np.arange(n_peaks, dtype=int),
                "peak_value_rel": peak_values_rel,
            }
        )
        df.to_csv(output_dir_timestamped / f"{group_id}.csv", index=False)

        # Optionally export as html figure
        if args.save_html_figure:
            plot_prompts_html(
                trajectories_xy,
                peaks_xy,
                peak_values_rel,
                bboxes_clipped_x1y1x2y2,
                group_id,
                video_length_minutes,
                args.image_width,
                args.image_height,
                args.peaks_threshold_rel,
                output_dir_timestamped / f"{group_id}.html",
            )

        # return peaks_xy.shape[0]
        print(f"Group {group_id} ({len(leaves)} videos): {n_peaks} prompts")

    print(f"Output written to {output_dir_timestamped}")


def parse_args(list_args: list[str]) -> argparse.Namespace:
    """Parse CLI args."""
    parser = argparse.ArgumentParser(
        description=(
            "Generate point and bbox prompts per video for "
            "segmenting burrows with SAM3"
            "from the trajectory data in the input "
            "zarr store."
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
            "Per-video CSVs (and HTML figures, if requested) are saved here."
        ),
    )
    parser.add_argument(
        "--bin-size-pixels",
        type=int,
        default=5,
        help=(
            "Bin edge length in pixels, for the trajectory 2D "
            "histogram (default: 5)."
        ),
    )
    parser.add_argument(
        "--counts-percentile",
        type=float,
        default=99,
        help=(
            "Histogram bins with counts below this percentile are zeroed "
            "before smoothing (default: 99). Values range from 0 to 100"
        ),
    )
    parser.add_argument(
        "--gaussian-sigma",
        type=float,
        default=2.5,
        help=(
            "Sigma (in bins) of the Gaussian filter applied to "
            "the log-transformed counts (default: 2.5)."
        ),
    )
    parser.add_argument(
        "--peaks-min-distance",
        type=int,
        default=10,
        help=(
            "Minimum separation (in bins) between two detected peaks, passed "
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
    parser.add_argument(
        "--group-by-pattern",
        action="store_true",
        help=(
            "If set, trajectory data comes from groups "
            "that match the provided "
            "pattern, rather than by video. Default: not set."
        ),
    )
    parser.add_argument(
        "--patterns",
        nargs="+",
        default=[
            "04.09.2023*",
            "05.09.2023*",
            "06.09.2023*",
            "07.09.2023*",
            "09.08.2023*",
            "10.08.2023*",
        ],
        help=(
            "If --group-by-pattern is set, this list of patterns is used. "
            "Patterns are matched against leaf node paths via "
            "DataTree.match. Note that if this argument is set without "
            "--group-by-pattern being passed, the patterns are ignored. "
            "Default: %(default)s"
        ),
    )

    args = parser.parse_args(list_args)

    # Check: --patterns only takes effect with --group-by-pattern
    # ATT: a user passing the exact default list explicitly wouldn't trigger
    # the check
    if (
        args.patterns != parser.get_default("patterns")
        and not args.group_by_pattern
    ):
        parser.error("--patterns requires --group-by-pattern")

    return args


if __name__ == "__main__":
    args = parse_args(sys.argv[1:])
    main(args)
