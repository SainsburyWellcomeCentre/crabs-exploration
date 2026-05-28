# %%
from datetime import datetime  # noqa: E402
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

# path to csv with candidate prompts derived from trajectory data
prompt_coords_dir = (
    project_dir / "burrow_prompts_per_day_20260423_143244"
    # project_dir / "burrow_prompts_per_video_20260423_153811"
)

# Select whether prompts are grouped by video or by date
flag_using_video_prompts = bool("video" in prompt_coords_dir.stem)

# path to trajectory dataset
crabs_zarr_dataset = (
    Path.home()
    / "swc"
    / "CrabTracks"
    / "CrabTracks-slurm2478780-2478861-2489356.zarr"
)


# Exemplars csv
OUTPUT_DIR = project_dir / "crabs-exploration" / "output_burrows_sam3"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


# Trajectory rasterisation params (datashader)
DYNSPREAD_THRESHOLD = 0.975
TRAJ_COLOR = "#c3ff1f"

# If the cache directory already exists, we skip rasterisation and load
# from disk
traj_cache_dir = project_dir / "burrow_trajectory_rasters"


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

# Empty points layer ready for red cross markers
# TODO: reduce line width
viewer.add_points(
    np.empty((0, 3)),
    ndim=3,
    name="manual points",
    symbol="x",
    face_color="transparent",
    border_color="red",
    size=35,
    border_width=0.1
)

# %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
# Export data in "manual points" layer as a csv.
# TODO: maybe save the point results to a zarr as I go, then export as csv?

# Get data from napari
# The points layer holds (z, y, x) coordinates where z is the frame index;
points_data = viewer.layers["manual points"].data  # (N, 3): z, y, x
frame_idx_per_point = (points_data[:, 0]).astype(int)

# Build dataframe
# group_id is set to the corresponding RGB image filename
df_manual_points = pd.DataFrame(
    {
        "group_id": [
            image_array.img_paths[i].name for i in frame_idx_per_point
        ],
        "prompt_point_x": points_data[:, 2],
        "prompt_point_y": points_data[:, 1],
    }
)

# Export as csv
timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
output_csv = OUTPUT_DIR / f"manual_prompt_points_{timestamp}.csv"
df_manual_points.to_csv(output_csv, index=False)
print(f"Saved {len(df_manual_points)} manual points to {output_csv}")

# %%
