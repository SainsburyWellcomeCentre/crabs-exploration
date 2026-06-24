# %%
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import xarray as xr
import zarr

# %%
# %matplotlib widget

# %%
# Parameters

min_samples_in_burrow_frac = 0.10

# %%
# Input data

# burrow masks
burrow_zarr = Path(
    "/Users/sofia/arc/project_Zoo_crabs/burrows/prelim_results_burrows/masks_pass_combined_20260619_160451.zarr"
    # TODO: video-first indexing at /home/sminano/swc/project_crabs/output_masks/masks_20260624_141224.zarr
)

# trajectory data
trajectories_zarr = Path(
    "/Users/sofia/swc/CrabTracks/CrabTracks-slurm2478780-2478861-2489356.zarr"
    # TODO: CrabTracks-slurm3012633.zarr
)

# %%
# Read trajectory data
dt = xr.open_datatree(trajectories_zarr, engine="zarr", chunks={})

# ds_video = dt[video_str].to_dataset()

# %%
# Read burrow data
burrows_zarr = zarr.open_group(burrow_zarr, mode="r")
burrows_masks = burrows_zarr["masks"]
burrows_scores = burrows_zarr["scores"]
# %%
# Get non-nan trajectory data samples and unique ID
video_str = "04.09.2023-01-Right"
ds_video = dt[video_str].to_dataset()
n_frames = ds_video.clip_last_frame_0idx.max().values.item() + 1

position_da = ds_video.position
x_da = position_da.sel(space="x")
y_da = position_da.sel(space="y")

# Keep only valid (non-nan) samples, then build the IDs only for those
x = x_da.values.reshape(-1)
y = y_da.values.reshape(-1)
valid = ~np.isnan(x) & ~np.isnan(y)
x_traj, y_traj = x[valid], y[valid]

# ----------------
# Compute traj-clip ids
# broadcast individuals/clip so the IDs match x and y element-by-element
trajectory_ids = (
    x_da["individuals"].broadcast_like(x_da).values.reshape(-1)[valid]
)  # can I avoid .values?
clip_ids = x_da["clip_id"].broadcast_like(x_da).values.reshape(-1)[valid]

# NOTE: trajectory IDs are reused across clips; to make a trajectory ID unique
# across video we combine it with clip_id. Factorizing each 1-D array (native
# dtype) and merging the codes avoids a slow row-wise unique over strings.
unique_traj_ids, traj_id_as_int = np.unique(
    trajectory_ids,
    return_inverse=True,
)
uniq_clip_ids, clip_id_as_int = np.unique(
    clip_ids,
    return_inverse=True,
)

# compute one integer per sample, representing a unique (traj_id, clip_id) pair
combined = np.ravel_multi_index(
    (traj_id_as_int, clip_id_as_int),
    (unique_traj_ids.size, uniq_clip_ids.size),
)

# integers from ravel_multi_index can have gaps; here we densify them
# using unique
unique_traj_clip_id, id_traj_clip = np.unique(combined, return_inverse=True)

del position_da, x, y, valid, trajectory_ids, clip_ids
del traj_id_as_int, clip_id_as_int, combined

# %%
# Compute subset of visited burrows
# TODO: to be changed to video-first indexing
burrow_id_mask = burrows_masks[0, :]
burrow_ids = np.unique(burrow_id_mask)

img_h, img_w = burrow_id_mask.shape
cols = np.round(x_traj).astype(int)
rows = np.round(y_traj).astype(int)
in_frame = (cols >= 0) & (cols < img_w) & (rows >= 0) & (rows < img_h)
traj_cols = cols[in_frame]
traj_rows = rows[in_frame]

hits_per_id = np.bincount(
    burrow_id_mask[traj_rows, traj_cols],  # masks IDs with traj data
    minlength=burrow_ids.shape[0],
)


# n_hits_excl_zero = np.sum(hits_per_id[1:] >= min_hits_per_burrow)

min_hits_per_burrow = int(min_samples_in_burrow_frac * n_frames)
visited_burrow_ids = np.argwhere(hits_per_id >= min_hits_per_burrow).reshape(
    -1
)
visited_burrow_ids = visited_burrow_ids[visited_burrow_ids != 0]

# %%
# plot visited burrows
# (excluding zero)
fig, ax = plt.subplots(1, 1)
ax.bar(np.arange(hits_per_id.shape[0])[1:], hits_per_id[1:])
ax.hlines(
    y=min_hits_per_burrow, xmin=0, xmax=hits_per_id.shape[0] + 1, colors="r"
)
ax.set_xlim([1, hits_per_id.shape[0] - 1])  # ok?
ax.set_xlabel("burrow ID")
ax.set_ylabel("trajectory samples")

# %%
# Get the trajectory-clip IDs that intersect each visited burrow
# TODO: right now a trajectory can be linked to several burrows
# (e.g. if it intersects a few). I need to fix this so that
# each trajectory is only fixed to one burrow (e.g. the one it sees
# the most?)

# burrow ID and trajectory ID for every in-frame sample
burrow_ID_per_traj_sample = burrow_id_mask[traj_rows, traj_cols]
traj_ids_per_sample = id_traj_clip[in_frame]

# map each selected burrow -> unique trajectory IDs that fall on it
visited_burrow_to_traj_ids = {
    burrow_id: np.unique(
        traj_ids_per_sample[burrow_ID_per_traj_sample == burrow_id]
    )
    for burrow_id in visited_burrow_ids
}









# %%
# plot
fig, ax = plt.subplots(1, 1)
ax.imshow(np.isin(burrow_id_mask, list(visited_burrow_to_traj_ids.keys())))

for burrow_id, traj_ids_arr in list(visited_burrow_to_traj_ids.items()):
    # fig, ax = plt.subplots(1, 1)
    # ax.imshow(burrow_id_mask==burrow_id)
    slc_trajectories = np.isin(id_traj_clip, traj_ids_arr)
    ax.scatter(x=x_traj[slc_trajectories], y=y_traj[slc_trajectories], s=0.5)

# %%
# Compute burrow centroids
from scipy.ndimage import center_of_mass

visited_burrow_centroids_yx = center_of_mass(
    np.ones(burrow_id_mask.shape, dtype=np.uint8),  # every pixel weight 1
    burrow_id_mask,
    index=visited_burrow_ids,
)
visited_burrow_centroids_xy = np.asarray(visited_burrow_centroids_yx)[:, ::-1]

visited_burrow_to_xy = {
    bid: visited_burrow_centroids_xy[k, :]
    for k, bid in enumerate(visited_burrow_to_traj_ids)
}

# %%
# plot burrow centroids
fig, ax = plt.subplots(1, 1)
ax.imshow(np.isin(burrow_id_mask, list(visited_burrow_to_traj_ids.keys())))

for burrow_xy in visited_burrow_centroids_xy:
    slc_trajectories = np.isin(id_traj_clip, traj_ids_arr)
    ax.scatter(x=burrow_xy[0], y=burrow_xy[1], s=15, marker="x")

# %%
# Compute vector from burrow centroid
vec_burrow_to_points = {}
for burrow_id, traj_ids_arr in list(visited_burrow_to_traj_ids.items()):
    slc_trajectories = np.isin(id_traj_clip, traj_ids_arr)
    vec_burrow_to_points[burrow_id] = (
        np.c_[x_traj[slc_trajectories], y_traj[slc_trajectories]]
        - visited_burrow_to_xy[burrow_id]
    )

# %%
# Compute distance to burrow centroid
# OJO!!
# A trajectory that passes through 3 burrows contributes its full sample
# set 3 times (once relative to each centroid).
# ---- How to fix that?

# TODO: normalise distance to average bbox size of trajectories linked
# to this burrow?
all_vec_burrow_to_points = np.concatenate(list(vec_burrow_to_points.values()))
fig, ax = plt.subplots()
ax.hist(np.linalg.norm(all_vec_burrow_to_points, axis=1))
ax.set_xlabel("pixels")
ax.set_ylabel("detections")  # --- can I express this as time...?

# Is it correct...? can I have 200_000 detections between 0 and 50 pixels of
# the centroid? the total video has 107_548 frames


# %%
theta = np.arctan2(
    all_vec_burrow_to_points[:, 1], all_vec_burrow_to_points[:, 0]
)

# %%
# Polar histogram of angles
n_bins = 36
counts, bin_edges = np.histogram(theta, bins=n_bins, range=(-np.pi, np.pi))
bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2
bin_width = bin_edges[1] - bin_edges[0]

fig, ax = plt.subplots(subplot_kw={"projection": "polar"})
ax.bar(bin_centers, counts, width=bin_width, bottom=0.0, align="center")
ax.set_theta_zero_location("E")
# ax.set_theta_offset(0)  # set zero at the right
ax.set_theta_direction(-1)  # theta increases in clockwise direction
ax.set_title("Angle of detections relative to burrow centroid")

# %%
