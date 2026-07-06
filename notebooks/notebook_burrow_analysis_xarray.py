# %%
from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import xarray as xr
import zarr
from matplotlib.lines import Line2D
from movement.kinematics import compute_speed, compute_velocity
from scipy.ndimage import center_of_mass

# %%
# %matplotlib qt
# qt / widget


# %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
# Input data

# burrow masks
burrow_zarr = Path(
    "/Users/sofia/arc/project_Zoo_crabs/burrows/prelim_results_burrows/masks_pass_combined_20260619_160451.zarr"
    # TODO: video-first indexing at /home/sminano/swc/project_crabs/output_masks/masks_20260624_141224.zarr
)

# trajectory data
trajectories_zarr = Path(
    "/Users/sofia/swc/CrabTracks/CrabTracks-slurm3012633.zarr"
    # CrabTracks-slurm2478780-2478861-2489356.zarr"
)

# for poster plots
raster_plots_dir = Path(
    "/Users/sofia/arc/project_Zoo_crabs/crabs-exploration/burrow_trajectory_rasters"
)

output_figs_dir = Path("/Users/sofia/arc/project_Zoo_crabs/ICN poster/figures")

# %%%%%%%%%%%%%
# Params

min_samples_in_burrow_frac = 0.10

# %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
# Read burrow data
burrows_zarr = zarr.open_group(burrow_zarr, mode="r")
burrows_masks = burrows_zarr["masks"]
burrows_scores = burrows_zarr["scores"]


# %%%%%%%%%%%%%%%%%%%%%%%%%%%%
# Read trajectory data
dt = xr.open_datatree(trajectories_zarr, engine="zarr", chunks={})

video_str = "04.09.2023-02-Right"
ds_video = dt[video_str].to_dataset()

n_frames_in_video = ds_video.clip_last_frame_0idx.max().values.item() + 1

# %%%%%%%%%%%%%%%%%%%%%%%%%%%
# Compute speed array
position_da = ds_video.position

velocity_da = compute_velocity(position_da)
speed_da = xr.apply_ufunc(
    np.linalg.norm,
    velocity_da,
    input_core_dims=[["space"]],
    kwargs={"axis": -1},
    dask="allowed",
)
speed_da.name = "speed"

ds_video["speed"] = speed_da

# %%%%%%%%%%%%%%%%%%%%
# Compute array of valid samples
valid_da = (
    ~position_da.sel(space="x").isnull() & ~position_da.sel(space="y").isnull()
)

# %%%%%%%%%%%%%%%%%%%
# Compute array of frame idcs
frame_in_clip_da = ds_video.time.broadcast_like(valid_da)
frame_in_clip_da = frame_in_clip_da.where(
    valid_da,
    -1,  # to keep as int
)
ds_video["frame_in_clip"] = frame_in_clip_da

frame_in_video_da = (ds_video.clip_first_frame_0idx + frame_in_clip_da).where(
    valid_da,
    -1,  # to keep as int
)
ds_video["frame_in_video"] = frame_in_video_da

print(ds_video.time.dtype)
# %%%%%%%%%%%%%%%%%%%%
# Combine traj data across all clips: compute traj-clip IDs
#
# clip_id and individuals are both dimensions of `position`, so a unique ID per
# (clip_id, individual) tracklet is just a dense 2-D array over those two axes.
# Individual labels are reused across clips; multiplying the clip index by the
# number of individuals makes the ID unique across the whole video (the xarray
# equivalent of ravel_multi_index, but trivial since these are already axes).
# It broadcasts over `time` on its own wherever it's combined with per-frame
# arrays.

# # NOTE: IDs are not dense!
# clip_idx_da = xr.DataArray(
#     np.arange(ds_video.sizes["clip_id"]), dims="clip_id"
# )
# indiv_idx_da = xr.DataArray(
#     np.arange(ds_video.sizes["individuals"]), dims="individuals"
# )
# traj_clip_id_da = clip_idx_da * ds_video.sizes["individuals"] + indiv_idx_da
# traj_clip_id_da.name = "traj_clip_id"

# ds_video["traj_clip_id"] = traj_clip_id_da


position_da = position_da.stack(traj_clip_id=("clip_id", "individuals"))

valid_da = valid_da.stack(traj_clip_id=("clip_id", "individuals"))

frame_in_video_da = frame_in_video_da.stack(
    traj_clip_id=("clip_id", "individuals")
)
# %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
# Compute subset of visited burrows
# TODO: to be changed when I move to zarr with video-first indexing
list_videos = [lv.name for lv in dt.leaves]
video_idx_zarr = np.argwhere(video_str == np.asarray(list_videos)).item()
# ------------------

burrow_id_mask = burrows_masks[video_idx_zarr, :]
img_h, img_w = burrow_id_mask.shape

# Compute pixels with trajectory data
# select pixel positions with valid position and within frame
cols_px = np.floor(position_da.sel(space="x").values).ravel()
rows_px = np.floor(position_da.sel(space="y").values).ravel()
pixels_to_keep = (
    valid_da.values.ravel()
    & ((cols_px >= 0) & (cols_px < img_w))
    & ((rows_px >= 0) & (rows_px < img_h))
)
cols_px = cols_px[pixels_to_keep].astype(int)
rows_px = rows_px[pixels_to_keep].astype(int)

# count how many trajectory samples (regardless of individual)
# intersect with each burrow
hits_per_id = np.bincount(
    burrow_id_mask[rows_px, cols_px],
    minlength=burrow_id_mask.max() + 1,
)

min_hits_per_burrow = min_samples_in_burrow_frac * n_frames_in_video
visited_burrow_ids = np.flatnonzero(hits_per_id >= min_hits_per_burrow)
visited_burrow_ids = visited_burrow_ids[visited_burrow_ids != 0]

# %%%%%%%%%%%%%%%%%%%%%%%%%%
# Compute visited burrow centroids
visited_burrow_centroids_yx = center_of_mass(
    np.ones(burrow_id_mask.shape, dtype=np.uint8),  # every pixel weight 1
    burrow_id_mask,
    index=visited_burrow_ids,
)
visited_burrow_centroids_xy = np.asarray(visited_burrow_centroids_yx)[:, ::-1]

visited_burrows_da = xr.DataArray(
    visited_burrow_centroids_xy,
    coords={"burrow_id": visited_burrow_ids, "space": ["x", "y"]},
)

# %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
# Assign trajectory-clip IDs to a single visited burrow
# IMPORTANT:
# - each trajectory is linked to only one burrow
# (the one it "sees" the most)

# traj_code == traj_clip_id densified
traj_code_da = xr.DataArray(
    np.arange(position_da.sizes["traj_clip_id"]),
    dims="traj_clip_id",
)

# compute traj_codes per non-nan sample
traj_code_per_sample = traj_code_da.broadcast_like(
    position_da.sel(space="x")
).values.ravel()[pixels_to_keep]


# compute samples in visited burrows
burrow_id_per_sample = burrow_id_mask[rows_px, cols_px]
in_visited_burrow = np.isin(burrow_id_per_sample, visited_burrow_ids)


# build count matrix
count_matrix = np.zeros(
    (
        position_da.sizes["traj_clip_id"],
        visited_burrows_da.sizes["burrow_id"],
    ),
    dtype=int,
)
np.add.at(
    count_matrix,
    (
        traj_code_per_sample[in_visited_burrow],
        np.searchsorted(
            visited_burrow_ids, burrow_id_per_sample[in_visited_burrow]
        ),  # index in visited_burrow_ids per sample
    ),
    1,
)

# link each trajectory to its most intersected burrow
# ties: lowest burrow ID
# traj_codes with no in-burrow samples get -1!
slc_traj_codes_no_in_burrow = count_matrix.sum(axis=1) == 0
burrow_per_traj_code = visited_burrow_ids[count_matrix.argmax(axis=1)]
burrow_per_traj_code[slc_traj_codes_no_in_burrow] = -1
# %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
# Resolve clashes in linked trajectories in time
# Two tracklets linked to the same burrow can't both exist at the same
# frame_in_video. Greedy: within each burrow, keep largest tracklets first,
# accepting one only if none of its valid frames are already taken.

n_samples_per_traj = valid_da.sum("time").values

# materialize frames for linked tracklets only (small subset of the 15810)
linked_traj_codes = np.flatnonzero(burrow_per_traj_code >= 0)
frame_in_video_linked = frame_in_video_da.isel(
    traj_clip_id=linked_traj_codes
).values
traj_code_to_column = {int(c): k for k, c in enumerate(linked_traj_codes)}

keep_traj_codes = np.zeros(position_da.sizes["traj_clip_id"], dtype=bool)
for b_id in visited_burrow_ids:
    # Get trajectories linked to this burrow sorted by length
    sorted_traj_codes_burrow = np.flatnonzero(burrow_per_traj_code == b_id)
    sorted_traj_codes_burrow = sorted_traj_codes_burrow[
        np.argsort(
            -n_samples_per_traj[sorted_traj_codes_burrow], kind="stable"
        )
        # kind=stable makes ties deterministic
    ]

    frame_occupied = np.zeros(n_frames_in_video, dtype=bool)
    for traj_code in sorted_traj_codes_burrow:
        # frames_in_video = frame_in_video_da.isel(traj_clip_id=traj_code).values  # ok?
        frames_in_video = frame_in_video_linked[
            :, traj_code_to_column[traj_code]
        ]
        frames_in_video = frames_in_video[frames_in_video != -1]
        if not frame_occupied[frames_in_video].any():
            keep_traj_codes[traj_code] = True
            frame_occupied[frames_in_video] = True

# Drop trajectories that should not be kept (set to -1)
burrow_per_traj_code = np.where(keep_traj_codes, burrow_per_traj_code, -1)

# %%
# Add burrow_id per linked trajectory as coord along
# traj_clip_id to position_da
position_da = position_da.assign_coords(
    burrow_id=(
        "traj_clip_id",  # coord
        burrow_per_traj_code,  # data
    )
)

# position_da.burrow_id.unstack("traj_clip_id")
# recovers it over (clip_id, individuals), with -1
# marking unlinked/dropped tracklets.
# %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
# Compute normalised position in burrow coord system (BCS) ----- REVIEW

# centroid of each tracklet's linked burrow. Keep the full traj_clip_id axis
# (15810): clamp -1 to a real id so .sel doesn't KeyError, then mask those
# entries back to NaN. Boolean-indexing here instead would shrink the axis to
# the linked subset and later align-intersect position_da down to it.
safe_burrow_id = position_da.burrow_id.where(
    position_da.burrow_id >= 0, visited_burrow_ids[0]
)
centroid_per_traj = (
    visited_burrows_da.sel(burrow_id=safe_burrow_id)
    .where(position_da.burrow_id >= 0)
    .transpose("space", "traj_clip_id")
)

# position in BCS (relative to centroid); broadcasts over time.
position_bcs_da = position_da - centroid_per_traj

# bbox diagonal per sample (~body length); NaN where no detection
bbox_shape_da = ds_video.shape.stack(traj_clip_id=("clip_id", "individuals"))
bbox_diag_da = np.hypot(
    bbox_shape_da.sel(space="x"),
    bbox_shape_da.sel(space="y"),
)

# ------------
# REVIEW
# normaliser: median bbox_diag pooled over time AND the burrow's kept tracklets.
# burrow_id is -1 for unlinked AND clash-dropped tracklets, so grouping by
# burrow_id automatically pools only the kept ones (the -1 group is ignored).
# Restrict to linked columns first to bound memory, then pool (time, traj) into
# one `sample` axis so the median reduces across both.
bbox_linked = bbox_diag_da.isel(traj_clip_id=linked_traj_codes)
burrow_linked = position_da.burrow_id.isel(traj_clip_id=linked_traj_codes)

bbox_np = bbox_linked.values  # (time, n_linked), NaN where invalid
burrow_np = burrow_linked.values  # (n_linked,)
burrow_flat = np.broadcast_to(burrow_np, bbox_np.shape).ravel()
bbox_flat = bbox_np.ravel()
m = ~np.isnan(bbox_flat)
median_bl_per_burrow = pd.Series(bbox_flat[m]).groupby(burrow_flat[m]).median()


# map the per-burrow factor back onto each tracklet, then normalise
# full-length (15810) burrow id per tracklet, -1 clamped to a real id for lookup
safe_burrow_id_full = position_da.burrow_id.where(
    position_da.burrow_id >= 0,
    visited_burrow_ids[0],  # WHY?
)

norm_per_traj = xr.DataArray(
    median_bl_per_burrow.reindex(safe_burrow_id_full.values).to_numpy(),
    dims="traj_clip_id",
    coords={"traj_clip_id": position_da.traj_clip_id},
).where(position_da.burrow_id >= 0)


position_bcs_bl_da = (
    position_bcs_da / norm_per_traj
)  # (space, time, traj_clip_id)
d_burrow_bl_da = np.hypot(
    position_bcs_bl_da.sel(space="x"),
    position_bcs_bl_da.sel(space="y"),
)  # (time, traj_clip_id)

# %%%%%%%%%%%%%%%%%%%
# Plot distance to burrow for selected burrow ID and color by speed
fps = float(ds_video.fps)

# speed was stored on ds_video before stacking, so stack it here.
speed_stacked_da = ds_video.speed.stack(
    traj_clip_id=("clip_id", "individuals")
)
for b_id in visited_burrow_ids:

    # tracklets linked to this burrow
    sel_traj = np.flatnonzero((position_da.burrow_id == b_id).values)

    # flatten the (time, traj_clip_id) subset; all three arrays share the layout,
    # so their ravel order matches element-for-element
    frame_arr = frame_in_video_da.isel(traj_clip_id=sel_traj).values.ravel()
    d_arr = d_burrow_bl_da.isel(traj_clip_id=sel_traj).values.ravel()
    speed_arr = speed_stacked_da.isel(traj_clip_id=sel_traj).values.ravel()

    # frame == -1 marks absent;
    mask = frame_arr != -1

    fig, ax = plt.subplots(figsize=(10, 10))
    sc = ax.scatter(
        x=frame_arr[mask] / fps / 60,
        y=d_arr[mask],
        c=speed_arr[mask] * fps,  # px/frame -> px/s (time coord is in frames)
        s=2.5,
        cmap="viridis",
        norm=mpl.colors.LogNorm(vmin=10),  # log colour scale (auto range from >0 data)
    )
    cbar = fig.colorbar(sc, ax=ax)
    cbar.set_label("speed (px/s)", fontsize=18)
    cbar.ax.tick_params(labelsize=16)


    for item in [ax.xaxis.label, ax.yaxis.label]:
        item.set_fontsize(20)
    ax.tick_params(axis="both", labelsize=18)
    ax.set_xlabel("time (min)")
    ax.set_ylabel(r"$\rho$ (BL)")
    ax.set_title(f"burrow ID {b_id}")
    fig.tight_layout()
    ax.spines[["top", "right"]].set_visible(False)

# %%
