"""Manually annotate burrow exemplar points over mean-frame images.

Launches an interactive napari viewer with one frame per video (or per day),
overlaid with:
* the rasterised crab trajectories of the corresponding video, and
* the candidate bbox prompts derived from the trajectory data
  (see compute_burrow_prompt_coords.py).

The user places point markers ("manual points") on burrow exemplars. The
points are autosaved to a timestamped CSV whenever they are added, removed or
moved, so an interrupted session never loses annotations. An existing CSV can
optionally be loaded at startup to continue a previous session.

The rasterised trajectories are cached as one RGBA PNG per video under
--traj-cache-dir. If that directory already exists, the cache is reused and the
zarr store is not read.

Output CSV columns (one row per manual point):
    group_id,        # the mean-frame image filename the point belongs to
    prompt_point_x,
    prompt_point_y

Usage (dependencies are auto-installed via uv):
* Annotate, with prompt CSVs grouped per video (default)
    uv run annotate_burrow_prompts_manual.py /path/to/images_dir \
        /path/to/prompt_coords_dir /path/to/store.zarr
* Prompt CSVs grouped per day
    uv run annotate_burrow_prompts_manual.py /path/to/images_dir \
        /path/to/prompt_coords_dir /path/to/store.zarr \
        --coord-prompts-grouped-by date
* Continue from a previous session
    uv run annotate_burrow_prompts_manual.py /path/to/images_dir \
        /path/to/prompt_coords_dir /path/to/store.zarr \
        --initial-points-csv /path/to/manual_prompt_points_*.csv
"""

# /// script
# requires-python = ">=3.11"
# dependencies = [
#   "napari",
#   "pyqt5",
#   "numpy>=2.0.0",
#   "pandas",
#   "xarray",
#   "dask",
#   "zarr",
#   "datashader",
#   "Pillow",
# ]
# ///
import argparse
import os
import sys
from datetime import datetime
from functools import partial
from pathlib import Path

import numpy as np
import pandas as pd
import xarray as xr
from PIL import Image


class ImageArrayLazy:
    """A lazy array for images in a list."""

    def __init__(self, img_paths):
        """Store sorted image paths and cache the shared image shape."""
        self.img_paths = sorted(img_paths)
        # add image shape, assuming all have same as
        # first sample
        sample = np.array(Image.open(img_paths[0]))  # H, W, C
        self.img_h, self.img_w, self.img_c = sample.shape

    def __len__(self):
        """Return the number of images."""
        return len(self.img_paths)

    def __getitem__(self, idx):
        """Load and return the image at ``idx`` as an array."""
        return np.array(Image.open(self.img_paths[idx]))

    @property
    def shape(self):
        """Return the stack shape (B, H, W, C)."""
        return (len(self.img_paths), self.img_h, self.img_w, self.img_c)
        # B, H, W, C


def _rasterise_video_trajectories(
    dt,
    video_str,
    canvas,
    img_h_i,
    img_w_i,
    traj_color,
    dynspread_th,
):
    """Return (H, W, 4) uint8 RGBA with the video's trajectories."""
    import datashader.transfer_functions as tf

    # Return zeros if video not in dataset
    if video_str not in dt:
        return np.zeros((img_h_i, img_w_i, 4), dtype=np.uint8)

    # Get non-nan x,y coords
    position = dt[video_str].to_dataset().position
    x = position.sel(space="x").values.reshape(-1)
    y = position.sel(space="y").values.reshape(-1)
    valid = ~np.isnan(x) & ~np.isnan(y)

    # Return zeros if no valid coords
    if not valid.any():
        return np.zeros((img_h_i, img_w_i, 4), dtype=np.uint8)

    # add data to canvas and rasterise
    agg = canvas.points(pd.DataFrame({"x": x[valid], "y": y[valid]}), "x", "y")
    shaded = tf.shade(agg, cmap=[traj_color])
    shaded = tf.dynspread(shaded, threshold=dynspread_th)

    # datashader y-origin is bottom; flip to match image y-origin (top)
    return np.array(shaded.to_pil().transpose(Image.FLIP_TOP_BOTTOM))


def _points_to_dataframe(points_data, img_paths):
    """Convert (N, 3) napari points (z, y, x) to saved CSV rows."""
    frame_idx_per_point = points_data[:, 0].astype(int)
    return pd.DataFrame(
        {
            "group_id": [img_paths[i].name for i in frame_idx_per_point],
            "prompt_point_x": points_data[:, 2],
            "prompt_point_y": points_data[:, 1],
        }
    )


def _dataframe_to_points(df, img_paths):
    """Convert saved CSV rows back to (N, 3) napari points (z, y, x)."""
    name_to_idx = {p.name: i for i, p in enumerate(img_paths)}
    z = df["group_id"].map(name_to_idx).to_numpy()
    return np.column_stack([z, df["prompt_point_y"], df["prompt_point_x"]])


def _autosave_manual_points(
    event=None,
    *,
    points_layer,
    img_paths,
    autosave_csv,
):
    """Save manual point labels to CSV (napari event callback).

    Atomic write via tmp + os.replace so a crash mid-write can't corrupt
    the previous good file. "Atomic" here means: at every moment, the file
    at autosave_csv is either the complete old version or the complete
    new version — never a half-written mix.
    """
    # skip the "adding", "removing" and changing intermediate events
    if event is not None and getattr(event, "action", None) not in (
        None,
        "added",
        "removed",
        "changed",
    ):
        return

    df = _points_to_dataframe(points_layer.data, img_paths)

    tmp_path = autosave_csv.with_suffix(".csv.tmp")
    df.to_csv(tmp_path, index=False)

    # saving and replacing is more robust to partial files if crashes;
    # replacing is instant, if to_csv fails we have old version
    os.replace(tmp_path, autosave_csv)


def _build_bbox_shapes(list_group_per_img, bboxes_xyxy_per_group):
    """Build napari shapes data for bbox prompts.

    A napari shapes layer expects each rectangle as a (4, 3) array of corners
    in (z, y, x) for a 3D viewer.
    """
    list_bbox_shapes = []
    for frame_idx, video_str in enumerate(list_group_per_img):
        bboxes = bboxes_xyxy_per_group.get(video_str)
        if bboxes is None:
            continue
        for x1, y1, x2, y2 in bboxes:
            list_bbox_shapes.append(
                np.array(
                    [
                        [frame_idx, y1, x1],
                        [frame_idx, y1, x2],
                        [frame_idx, y2, x2],
                        [frame_idx, y2, x1],
                    ],
                    dtype=float,
                )
            )
    return list_bbox_shapes


def _rasterise_and_cache_trajectories(
    zarr_store,
    traj_cache_dir,
    list_video_per_img,
    img_h,
    img_w,
    traj_color,
    dynspread_threshold,
):
    """Rasterise per-video trajectories and cache one RGBA PNG per video.

    Frames are computed per video, so we show the trajectories per video too.
    For each frame we rasterise the trajectories of its corresponding video
    onto a transparent canvas matching the frame resolution.
    """
    import datashader as ds

    # Read dataset
    dt = xr.open_datatree(zarr_store, engine="zarr", chunks={})

    # create output dir
    traj_cache_dir.mkdir(parents=True)

    # Prepare canvas
    canvas = ds.Canvas(
        plot_width=img_w,
        plot_height=img_h,
        x_range=(0, img_w),
        y_range=(0, img_h),
    )

    for video_str in list_video_per_img:
        # rasterise
        video_traj_array = _rasterise_video_trajectories(
            dt,
            video_str,
            canvas,
            img_h,
            img_w,
            traj_color,
            dynspread_threshold,
        )
        # save as png
        Image.fromarray(video_traj_array, mode="RGBA").save(
            traj_cache_dir / f"{video_str}.png"
        )


def main(args: argparse.Namespace) -> None:
    """Launch the interactive napari viewer for manual burrow annotation."""
    import napari

    flag_using_video_prompts = args.coord_prompts_grouped_by == "video"

    # Set up output dir
    args.output_dir.mkdir(parents=True, exist_ok=True)

    # ------------------------------------------------------------------
    # Read frames as a lazy array and map each frame to its video / date
    list_image_files = sorted(list(Path(args.images_dir).glob("*.png")))
    image_array = ImageArrayLazy(list_image_files)

    list_video_per_img = [
        img.stem.split("_", 1)[0].split("-Loop")[0]
        for img in image_array.img_paths
    ]
    list_date_per_img = [video.split("-")[0] for video in list_video_per_img]

    # Per-frame group string used to look up prompts / trajectories
    list_group_per_img = (
        list_video_per_img if flag_using_video_prompts else list_date_per_img
    )

    # ------------------------------------------------------------------
    # Read prompt data (one CSV per group, concatenated)
    list_prompt_csv = sorted(list(Path(args.prompt_coords_dir).glob("*.csv")))
    df_prompts = pd.concat([pd.read_csv(f) for f in list_prompt_csv])

    # bbox prompts in pixel xyxy, keyed by group_id
    bboxes_xyxy_per_group = {
        key: group[
            [
                "prompt_bbox_xmin",
                "prompt_bbox_ymin",
                "prompt_bbox_xmax",
                "prompt_bbox_ymax",
            ]
        ].to_numpy()
        for key, group in df_prompts.groupby("group_id")
    }

    # Build napari shapes data for bbox prompts
    list_bbox_shapes = _build_bbox_shapes(
        list_group_per_img, bboxes_xyxy_per_group
    )

    # ------------------------------------------------------------------
    # Build a per-video RGBA trajectory image stack.
    # If the cache does not exist, generate rasters and save.
    if not args.traj_cache_dir.exists():
        _rasterise_and_cache_trajectories(
            args.zarr_store,
            args.traj_cache_dir,
            list_video_per_img,
            image_array.img_h,
            image_array.img_w,
            args.traj_color,
            args.dynspread_threshold,
        )

    # load array from saved data
    # (ImageArrayLazy sorts image filenames alphabetically!)
    traj_paths = [args.traj_cache_dir / f"{v}.png" for v in list_video_per_img]
    traj_array = ImageArrayLazy(traj_paths)

    # ------------------------------------------------------------------
    # Set up path for saving manual labels
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_csv = args.output_dir / f"manual_prompt_points_{timestamp}.csv"

    # ------------------------------------------------------------------
    # Load all layers in napari
    viewer = napari.Viewer()

    # RGB frames as an image layer
    viewer.add_image(np.asarray(image_array), name="frames", rgb=True)

    # Trajectories as a transparent RGBA image layer aligned with the frames
    viewer.add_image(np.asarray(traj_array), name="trajectories", rgb=True)

    # Bbox prompts as a shapes layer
    viewer.add_shapes(
        list_bbox_shapes,
        shape_type="rectangle",
        edge_color="lime",
        face_color="transparent",
        edge_width=2,
        name="bbox prompts",
    )

    # Points layer ready for red cross markers.
    # Optionally pre-populated from a previously saved CSV.
    if (
        args.initial_points_csv is not None
        and args.initial_points_csv.exists()
    ):
        initial_points = _dataframe_to_points(
            pd.read_csv(args.initial_points_csv),
            image_array.img_paths,
        )
    else:
        initial_points = np.empty((0, 3))

    viewer.add_points(
        initial_points,
        ndim=3,
        name="manual points",
        symbol="x",
        face_color="transparent",
        border_color="red",
        size=35,
    )

    # Autosave "manual points" to CSV whenever points are added, removed
    # or moved
    viewer.layers["manual points"].events.data.connect(
        partial(
            _autosave_manual_points,
            points_layer=viewer.layers["manual points"],
            img_paths=image_array.img_paths,
            autosave_csv=output_csv,
        )
    )

    # Set points layer as active and in "Add" mode
    viewer.layers.selection.active = viewer.layers["manual points"]
    viewer.layers["manual points"].mode = "add"

    # Set slider to frame 0
    # First arg is the axis index, second is the step value
    viewer.dims.set_current_step(0, 0)

    print(f"Autosaving manual points to {output_csv}")

    # Start the Qt event loop and block until the viewer is closed
    napari.run()


def parse_args(list_args: list[str]) -> argparse.Namespace:
    """Parse CLI args."""
    parser = argparse.ArgumentParser(
        description=(
            "Interactively annotate burrow exemplar points over mean-frame "
            "images in napari, overlaid with rasterised trajectories and "
            "candidate bbox prompts. Points are autosaved to a timestamped "
            "CSV."
        ),
    )
    parser.add_argument(
        "images_dir",
        type=Path,
        help=(
            "Directory of mean-frame PNG images, one per video. The video "
            "(or date) each frame belongs to is parsed from its filename."
        ),
    )
    parser.add_argument(
        "prompt_coords_dir",
        type=Path,
        help=(
            "Directory of candidate bbox-prompt CSVs (one per group), as "
            "produced by compute_burrow_prompt_coords.py."
        ),
    )
    parser.add_argument(
        "zarr_store",
        type=Path,
        help=(
            "Path to the input trajectories zarr store, used to rasterise "
            "the per-video trajectory overlays. Only read if --traj-cache-dir "
            "does not already exist. Usually a CrabTracks zarr file produced "
            "by create-zarr-dataset."
        ),
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path.cwd() / "burrow_manual_exemplars",
        help=(
            "Directory for the autosaved manual-points CSV (default: "
            "./burrow_manual_exemplars). Created if it does not exist."
        ),
    )
    parser.add_argument(
        "--traj-cache-dir",
        type=Path,
        default=Path.cwd() / "burrow_trajectory_rasters",
        help=(
            "Directory for the cached per-video trajectory RGBA PNGs "
            "(default: ./burrow_trajectory_rasters). If it already exists, "
            "the cache is reused and the zarr store is not read."
        ),
    )
    parser.add_argument(
        "--initial-points-csv",
        type=Path,
        default=None,
        help=(
            "Optional CSV of previously saved manual points to load at "
            "startup, to continue a previous session. Default: start empty."
        ),
    )
    parser.add_argument(
        "--coord-prompts-grouped-by",
        choices=["video", "date"],
        default="video",
        help=(
            "Whether the candidate bbox-prompt CSVs (and trajectory lookups) "
            "are grouped per video or per date. Must match how "
            "prompt_coords_dir was generated (default: video)."
        ),
    )
    parser.add_argument(
        "--traj-color",
        type=str,
        default="#c3ff1f",
        help="Color of the rasterised trajectories (default: #c3ff1f).",
    )
    parser.add_argument(
        "--dynspread-threshold",
        type=float,
        default=0.975,
        help=(
            "datashader dynspread threshold for the rasterised trajectories "
            "(default: 0.975)."
        ),
    )

    return parser.parse_args(list_args)


if __name__ == "__main__":
    args = parse_args(sys.argv[1:])
    main(args)
