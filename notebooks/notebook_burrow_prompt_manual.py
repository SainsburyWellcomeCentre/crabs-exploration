# %%
import os
from datetime import datetime  # noqa: E402
from functools import partial
from pathlib import Path

import datashader as ds
import datashader.transfer_functions as tf
import napari
import numpy as np
import pandas as pd
import xarray as xr
from PIL import Image

# %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
# Input data
project_dir = Path("/Users/sofia/arc/project_Zoo_crabs/")

# mean frames per video
images_dir = project_dir / "burrow_mean_image_slurm_3014447"

# path to trajectory dataset
crabs_zarr_dataset = (
    Path.home()
    / "swc"
    / "CrabTracks"
    / "CrabTracks-slurm2478780-2478861-2489356.zarr"
)

# -----------
# path to csv with candidate bbox prompts derived from trajectory data
prompt_coords_dir = (
    # project_dir / "burrow_prompts_per_day_20260423_143244"
    project_dir / "burrow_prompts_per_video_20260423_153811"
)

# Select whether candidate bbox prompts are grouped by video or by date
flag_using_video_prompts = bool("video" in prompt_coords_dir.stem)

# -----------
# Trajectory rasterisation params (datashader)
DYNSPREAD_THRESHOLD = 0.975
TRAJ_COLOR = "#c3ff1f"

# If the cache directory already exists, we skip rasterisation and load
# from disk
traj_cache_dir = project_dir / "burrow_trajectory_rasters"

# -----------
# Directory for output csv (autosaved)
OUTPUT_DIR = project_dir / "burrow_manual_exemplars"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# Optional: load existing manual annotations at startup. Set to None to start empty.
initial_points_csv: Path | None = Path(
    "/Users/sofia/arc/project_Zoo_crabs/crabs-exploration/burrow_manual_exemplars/manual_prompt_points_20260528_154656.csv"
)


# %%%%%%%%%%
# Helpers
class ImageArrayLazy:
    """A lazy array for images in a list."""

    def __init__(self, img_paths):
        self.img_paths = sorted(img_paths)
        # add image shape, assuming all have same as
        # first sample
        sample = np.array(Image.open(img_paths[0]))  # H, W, C
        self.img_h, self.img_w, self.img_c = sample.shape

    def __len__(self):
        return len(self.img_paths)

    def __getitem__(self, idx):
        return np.array(Image.open(self.img_paths[idx]))

    @property
    def shape(self):
        return (len(self.img_paths), self.img_h, self.img_w, self.img_c)
        # B, H, W, C


def _rasterise_video_trajectories(
    dt,
    video_str,
    canvas,
    img_h_i,
    img_w_i,
    traj_color=TRAJ_COLOR,
    dynspread_th=DYNSPREAD_THRESHOLD,
):
    """Return (H, W, 4) uint8 RGBA with the video's trajectories."""
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


def _points_to_dataframe(points_data):
    frame_idx_per_point = points_data[:, 0].astype(int)
    return pd.DataFrame(
        {
            "group_id": [
                image_array.img_paths[i].name for i in frame_idx_per_point
            ],
            "prompt_point_x": points_data[:, 2],
            "prompt_point_y": points_data[:, 1],
        }
    )


def _dataframe_to_points(df, img_paths):
    """Convert saved CSV rows back to (N, 3) napari points (z, y, x)."""
    name_to_idx = {p.name: i for i, p in enumerate(img_paths)}
    z = df["group_id"].map(name_to_idx).to_numpy()
    return np.column_stack([z, df["prompt_point_y"], df["prompt_point_x"]])


def _autosave_manual_points(autosave_csv, event=None):
    """Define callback for saving manual point labels.

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

    points_data = viewer.layers["manual points"].data
    df = _points_to_dataframe(points_data)

    tmp_path = autosave_csv.with_suffix(".csv.tmp")
    df.to_csv(tmp_path, index=False)

    # saving and replacing is more robust to partial files if crashes;
    # replacing is instant, if to_csv fails we have old version
    os.replace(tmp_path, autosave_csv)


# %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
# Read frames as a lazy array and map each frame to its video / date
list_image_files = sorted(list(Path(images_dir).glob("*.png")))
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


# %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
# Read prompt data (one CSV per group, concatenated)
list_prompt_csv = sorted(list(Path(prompt_coords_dir).glob("*.csv")))
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

# %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
# Build napari shapes data for bbox prompts.
# napari shapes layer expects each rectangle as a (2, 3) array of opposite
# corners in (z, y, x) for a 3D viewer.
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

# %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
# Build a per-video RGBA trajectory image stack.
# Frames are computed per video, so we show the trajectories per video too.
# For each frame we rasterise the trajectories of its corresponding video
# onto a transparent canvas matching the frame resolution.

# If rasters do not exist, generate rasters and save
if not traj_cache_dir.exists():
    # Read dataset
    dt = xr.open_datatree(crabs_zarr_dataset, engine="zarr", chunks={})

    # create output dir
    traj_cache_dir.mkdir(parents=True)

    # Prepare canvas
    img_h_i, img_w_i = image_array.img_h, image_array.img_w
    canvas = ds.Canvas(
        plot_width=img_w_i,
        plot_height=img_h_i,
        x_range=(0, img_w_i),
        y_range=(0, img_h_i),
    )

    for video_str in list_video_per_img:
        # rasterise
        video_traj_array = _rasterise_video_trajectories(
            dt,
            video_str,
            canvas,
            img_h_i,
            img_w_i,
        )
        # save as png
        Image.fromarray(video_traj_array, mode="RGBA").save(
            traj_cache_dir / f"{video_str}.png"
        )

# %%
# load array from saved data
traj_paths = [traj_cache_dir / f"{v}.png" for v in list_video_per_img]
traj_array = ImageArrayLazy(
    traj_paths
)  # sorts image filenames alphabetically!

# %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
# Set up path for saving manual labels
timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
output_csv = OUTPUT_DIR / f"manual_prompt_points_{timestamp}.csv"

# %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
# Load all layers in napari
viewer = napari.Viewer()

# RGB frames as an image layer
viewer.add_image(np.asarray(image_array), name="frames", rgb=True)

# Trajectories as a transparent RGBA image layer aligned with the frames
viewer.add_image(np.asarray(traj_array), name="trajectories", rgb=True)

# Bbox prompts (computed per-day) as a shapes layer
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
if initial_points_csv is not None and initial_points_csv.exists():
    initial_points = _dataframe_to_points(
        pd.read_csv(initial_points_csv),
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

# Set up autosave "manual points" to CSV
# whenever points are added, removed or moved
viewer.layers["manual points"].events.data.connect(
    partial(_autosave_manual_points, autosave_csv=output_csv)
)

# %%
# Set points layer as active and in "Add" mode
viewer.layers.selection.active = viewer.layers["manual points"]
viewer.layers["manual points"].mode = "add"

# Set slider to frame 0
# First arg is the axis index, second is the step value
viewer.dims.set_current_step(0, 0)

# %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
# Export data in "manual points" layer as a csv.

# # Get data from napari
# # The points layer holds (z, y, x) coordinates where z is the frame index;
# points_data = viewer.layers["manual points"].data  # (N, 3): z, y, x
# frame_idx_per_point = (points_data[:, 0]).astype(int)

# # Build dataframe
# # group_id is set to the corresponding RGB image filename
# df_manual_points = _points_to_dataframe(points_data)

# # Export as csv
# timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
# output_csv = OUTPUT_DIR / f"manual_prompt_points_{timestamp}.csv"
# df_manual_points.to_csv(output_csv, index=False)
# print(f"Saved {len(df_manual_points)} manual points to {output_csv}")

# %%
