# %%
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import xarray as xr
import zarr
from scipy.ndimage import center_of_mass

# %%
%matplotlib widget

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
    "/Users/sofia/swc/CrabTracks/CrabTracks-slurm3012633.zarr" 
    #CrabTracks-slurm2478780-2478861-2489356.zarr"
)

# %%
def factorize_string_coord_in_da(da, coord_name): 
    coord_da = da[coord_name]
    coords_factorized, coords_unique = pd.factorize(coord_da.values)
    return coord_da.copy(data=coords_factorized), coords_unique   # same dims, int data


# TODO: review!
# def flat_valid(input_da, ref_da, valid):
#     # align the 1-D coord to x_da's dims, broadcast as a VIEW, then mask
#     input_da_broadcasted = input_da.broadcast_like(ref_da)  # keeps it lazy/strided
#     return np.broadcast_to(input_da_broadcasted.values, ref_da.shape).reshape(-1)[valid]


# %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
# Read burrow data
burrows_zarr = zarr.open_group(burrow_zarr, mode="r")
burrows_masks = burrows_zarr["masks"]
burrows_scores = burrows_zarr["scores"]


# %%%%%%%%%%%%%%%%%%%%%%%%%%%%
# Read trajectory data
dt = xr.open_datatree(trajectories_zarr, engine="zarr", chunks={})

video_str = "04.09.2023-01-Right"
ds_video = dt[video_str].to_dataset()

# %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
# Get non-nan trajectory data samples
n_frames = ds_video.clip_last_frame_0idx.max().values.item() + 1

position_da = ds_video.position
x_da = position_da.sel(space="x")
y_da = position_da.sel(space="y")

# Keep only valid (non-nan) samples, then build the IDs only for those
x = x_da.values.reshape(-1)
y = y_da.values.reshape(-1)
valid = ~np.isnan(x) & ~np.isnan(y)
x_traj, y_traj = x[valid], y[valid]

# %%
# Free from memory
del position_da, x, y


# %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
# Compute traj-clip ids

indiv_code_da, unique_indivs = factorize_string_coord_in_da(x_da, "individuals")
clip_code_da, uniq_clip_ids = factorize_string_coord_in_da(x_da,"clip_id")

# broadcast individuals/clip so the IDs match x and y element-by-element
# (we broadcast ints rather than str)
indiv_id_as_int = indiv_code_da.broadcast_like(x_da).values.reshape(-1)[valid]
clip_id_as_int = clip_code_da.broadcast_like(x_da).values.reshape(-1)[valid]

# NOTE: trajectory IDs are reused across clips; to make a trajectory ID unique
# across video we combine it with clip_id. Factorizing each 1-D array (native
# dtype) and merging the codes avoids a slow row-wise unique over strings.

# compute one integer per sample, representing a unique (traj_id, clip_id) pair
combined = np.ravel_multi_index(
    (indiv_id_as_int, clip_id_as_int),
    (unique_indivs.size, uniq_clip_ids.size),
)

# integers from ravel_multi_index can have gaps; here we densify them
# using unique
# (traj_clip_id_dense_per_sample holds the indices of _unique_traj_clip_id;
# we use it as a dense traj_clip_ID)
_unique_traj_clip_id, traj_clip_id_dense_per_sample = np.unique(
    combined, return_inverse=True
)
unique_traj_clip_id_dense = np.arange(traj_clip_id_dense_per_sample.max() + 1)
# same as np.unique(traj_clip_id_dense_per_sample)

# %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
# Compute frame per sample
# 0-based indexing
frame_in_clip_per_sample = (
    x_da["time"].broadcast_like(x_da).values.reshape(-1)[valid]
) 

frame_in_video_per_sample = (
    frame_in_clip_per_sample
    + x_da["clip_first_frame_0idx"]
    .broadcast_like(x_da)
    .values.reshape(-1)[valid]
)
# %%
# Free from memory
del valid, unique_indivs, uniq_clip_ids
del indiv_id_as_int, clip_id_as_int, combined

# %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
# Compute subset of visited burrows
# TODO: to be changed when I move to zarr with video-first indexing
list_videos = [lv.name for lv in dt.leaves]
video_idx_zarr = np.argwhere(video_str==np.asarray(list_videos)).item()
# ------------------

burrow_id_mask = burrows_masks[video_idx_zarr, :] 
burrow_ids = np.unique(burrow_id_mask)

img_h, img_w = burrow_id_mask.shape
cols = np.round(x_traj).astype(int)  # floor?
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


# %%%%%%%%%%%%%%%%%%%%%%%%%%
# Compute visited burrow centroids
visited_burrow_centroids_yx = center_of_mass(
    np.ones(burrow_id_mask.shape, dtype=np.uint8),  # every pixel weight 1
    burrow_id_mask,
    index=visited_burrow_ids,
)
visited_burrow_centroids_xy = np.asarray(visited_burrow_centroids_yx)[:, ::-1]

visited_burrow_to_xy = {
    bid: visited_burrow_centroids_xy[k, :]
    for k, bid in enumerate(visited_burrow_ids)
}


# %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
# Assign trajectory-clip IDs to a single visited burrow
# IMPORTANT: each trajectory is only fixed to one burrow
# (the one it "sees" the most)

# get burrow ID and trajectory ID for every in-frame sample
burrow_ID_per_sample = burrow_id_mask[traj_rows, traj_cols]
traj_ids_per_sample = traj_clip_id_dense_per_sample[in_frame]

# Build a matrix of shape (traj_ids, visited_burrow_ids)
# to keep track of their intersections
count_matrix = np.zeros(
    (unique_traj_clip_id_dense.shape[0], visited_burrow_ids.shape[0]),
    dtype=int,
)
for b_id, burrow_id in enumerate(visited_burrow_ids):
    dense_traj_ids, counts = np.unique(
        traj_ids_per_sample[burrow_ID_per_sample == burrow_id],
        return_counts=True,
    )
    count_matrix[dense_traj_ids, b_id] = counts

# %%
# Reduce count matrix to consider visited traj IDs only
# (this is because argmax of row all 0s will return 0)
slc_rows = count_matrix.sum(axis=1) > 0
visited_traj_ids = unique_traj_clip_id_dense[slc_rows]
count_matrix_reduced = count_matrix[slc_rows, :]

# Link each visited traj ID to the burrow it maximally
# intersects with
# NOTE: with argmax ties resolve to lowest column index
burrow_id_per_visited_traj_id = visited_burrow_ids[
    np.argmax(count_matrix_reduced, axis=1)
]

# Build dict mapping burrow ID to linked traj IDs
visited_burrow_to_traj_ids = {}
for burrow_id in visited_burrow_ids:
    visited_burrow_to_traj_ids[burrow_id] = visited_traj_ids[
        burrow_id_per_visited_traj_id == burrow_id
    ]


# %%
# Check: each trajectory ID is linked to at most one burrow
all_linked_traj_ids = np.concatenate(list(visited_burrow_to_traj_ids.values()))
assert all_linked_traj_ids.size == np.unique(all_linked_traj_ids).size, (
    "Some trajectory ID is linked to more than one burrow"
)


# %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
# Build the per-sample dataframe
#
# One row per valid (non-nan) trajectory sample. We map each sample to the
# burrow its trajectory is linked to (NA if the trajectory isn't linked to
# any visited burrow), look up that burrow's centroid, and express the sample
# position relative to it. All per-sample arrays were already extracted in a
# memory-efficient way above (lazy read + broadcast ints + valid mask), so
# here we only assemble cheap lookups indexed by the dense traj-clip ID.

# map dense traj-clip ID -> linked burrow ID and centroid (default: unlinked)
n_traj_clip = unique_traj_clip_id_dense.shape[0]
burrow_id_per_traj = np.full(n_traj_clip, -1, dtype=int)
centroid_xy_per_traj = np.full((n_traj_clip, 2), np.nan)
burrow_id_per_traj[visited_traj_ids] = burrow_id_per_visited_traj_id
for burrow_id, traj_ids_arr in visited_burrow_to_traj_ids.items():
    centroid_xy_per_traj[traj_ids_arr] = visited_burrow_to_xy[burrow_id]

# expand the per-traj lookups to one value per sample
burrow_id_per_sample = burrow_id_per_traj[traj_clip_id_dense_per_sample]
centroid_xy_per_sample = centroid_xy_per_traj[traj_clip_id_dense_per_sample]

df = pd.DataFrame(
    {
        "sample_idx": np.arange(x_traj.shape[0]),
        "x": x_traj,
        "y": y_traj,
        "traj_clip_id": traj_clip_id_dense_per_sample,
        # burrow_id is nullable (NA for samples not linked to any burrow)
        "burrow_id": pd.arrays.IntegerArray(
            burrow_id_per_sample.astype("int64"),
            mask=burrow_id_per_sample < 0,
        ),
        "burrow_centroid_x": centroid_xy_per_sample[:, 0],
        "burrow_centroid_y": centroid_xy_per_sample[:, 1],
        "x_burrow": x_traj - centroid_xy_per_sample[:, 0],
        "y_burrow": y_traj - centroid_xy_per_sample[:, 1],
        "frame_in_clip": frame_in_clip_per_sample,
        "frame_in_video": frame_in_video_per_sample,
    }
)

# samples whose trajectory is linked to a visited burrow (used by most plots)
df_linked = df.dropna(subset=["burrow_id"])

# %%
# Free from memory
# (per-sample arrays now live in the dataframe)
del x_traj, y_traj, traj_clip_id_dense_per_sample
del frame_in_clip_per_sample, frame_in_video_per_sample
del burrow_id_per_sample, centroid_xy_per_sample
del burrow_id_per_traj, centroid_xy_per_traj

# %%
# plot visited burrows
# (excluding zero)
# NOTE: hits_per_id counts every in-frame sample falling on each burrow mask
# (before trajectories are assigned to a single burrow), so it can't be derived
# from the dataframe; we only read the selected-burrow count from df_linked
fig, ax = plt.subplots(1, 1)
ax.bar(np.arange(hits_per_id.shape[0])[1:], hits_per_id[1:])
ax.hlines(
    y=min_hits_per_burrow, xmin=0, xmax=hits_per_id.shape[0] + 1, colors="r"
)
ax.set_xlim([1, hits_per_id.shape[0] - 1])  # ok?
ax.set_xlabel("burrow ID")
ax.set_ylabel("trajectory samples")
ax.set_title(f"n = {df_linked['burrow_id'].nunique()} selected burrows")

# %%
# plot burrow centroids
# (one centroid per visited burrow, read from the dataframe)
burrow_centroids = df_linked.groupby("burrow_id")[
    ["burrow_centroid_x", "burrow_centroid_y"]
].first()

fig, ax = plt.subplots(1, 1)
ax.imshow(np.isin(burrow_id_mask, burrow_centroids.index.to_numpy("int64")))

for _, (cx, cy) in burrow_centroids.iterrows():
    ax.scatter(x=cx, y=cy, s=15, marker="x")


# %%
# plot burrows and their linked trajectories
fig, ax = plt.subplots(1, 1)
ax.imshow(np.zeros_like(burrow_id_mask))  # , cmap="gray")

cmap = plt.get_cmap("tab20")
for k, (burrow_id, group) in enumerate(df_linked.groupby("burrow_id")):
    color = cmap(k % cmap.N)
    ax.scatter(
        x=group["x"],
        y=group["y"],
        s=0.5,
        color=color,
    )
    ax.contour(
        burrow_id_mask == burrow_id,
        levels=[0.5],
        colors=[color],
        linewidths=1,
    )


# %%
# plot all trajectories with transparency with burrow centroid at the origin
# (position relative to the burrow centroid lives in df["x_burrow"/"y_burrow"])
fig, ax = plt.subplots(1, 1)

cmap = plt.get_cmap("tab20")
ax.scatter(
    x=df_linked["x_burrow"],
    y=df_linked["y_burrow"],
    s=0.5,
    color=cmap(0),
    alpha=0.05,
)

ax.scatter(x=0, y=0, s=30, marker="x", color="k", zorder=5)

# rings at radial percentiles: each circle encloses a given fraction of
# detections, so closely-spaced rings indicate high density
ring_percentiles = [50, 75, 95, 100]
all_radii = np.hypot(df_linked["x_burrow"], df_linked["y_burrow"])
ring_radii = np.percentile(all_radii, ring_percentiles)
for pct, radius in zip(ring_percentiles, ring_radii, strict=True):
    ax.add_patch(
        plt.Circle(
            (0, 0),
            radius,
            fill=False,
            edgecolor="k",
            linewidth=0.5,
            linestyle="--",
            alpha=0.5,
            zorder=4,
        )
    )
    ax.annotate(
        f"{pct}% ({radius:.0f} px)",
        xy=(0, -radius),
        ha="center",
        va="bottom",
        fontsize=7,
        color="k",
        zorder=6,
    )

ax.set_aspect("equal")
ax.invert_yaxis()  # match image coordinates (y down)
ax.set_xlabel("$x_{burrow}$ (pixels)")
ax.set_ylabel("$y_{burrow}$ (pixels)")
ax.set_title("Trajectories in burrow coord syst")

# %%
# density (2D histogram) of points relative to the burrow centroid
# all_vec_burrow_to_points = np.concatenate(list(vec_burrow_to_points.values()))

# bin_size = 200  # pixels
# half_extent = 2000  # pixels, square window around the centroid
# bin_edges = np.arange(-half_extent, half_extent + bin_size, bin_size)

# fig, ax = plt.subplots(1, 1)
# _, _, _, im = ax.hist2d(
#     all_vec_burrow_to_points[:, 0],
#     all_vec_burrow_to_points[:, 1],
#     bins=[bin_edges, bin_edges],
#     cmin=1,  # leave empty bins blank
# )
# fig.colorbar(im, ax=ax, label="detections")

# ax.scatter(x=0, y=0, s=30, marker="x", color="r", zorder=5)
# ax.set_aspect("equal")
# ax.invert_yaxis()  # match image coordinates (y down)
# ax.set_xlabel("$x_{burrow}$ (pixels)")
# ax.set_ylabel("$y_{burrow}$ (pixels)")
# ax.set_title(f"Detection density ({bin_size} px bins)")


# %%%%%%%%%%%%%%
# Compute histogram of distance to burrow centroid

# TODO: normalise distance to average bbox size of trajectories linked
# to this burrow?
fig, ax = plt.subplots()
ax.hist(all_radii)
ax.set_xlabel("pixels")
ax.set_ylabel("detections")  # --- can I express this as time...?


# # single burrow
# single = df_linked[df_linked["burrow_id"] == 30]
# fig, ax = plt.subplots()
# ax.hist(np.hypot(single["x_burrow"], single["y_burrow"]))
# ax.set_xlabel("pixels")
# ax.set_ylabel("detections")


# %%%%%%%%%%%%%%%%%%%%%%%%%%
# Compute theta angle relative to x-axis in BCS
theta = np.arctan2(df_linked["y_burrow"], df_linked["x_burrow"])

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

# %%%%%%%%%%%%%%%%%%%%%%%
# Compute distance to burrow in time
for b_id, group in df_linked.groupby("burrow_id"):
    # sel_burrow_id = 30

    d_burrow = np.hypot(group["x_burrow"], group["y_burrow"])

    # plot
    fig, (ax, ax_traj) = plt.subplots(1, 2, figsize=(12, 5))

    # left: distance to burrow over time, coloured by trajectory ID
    # TODO: mark when in burrow?
    cmap = plt.get_cmap("tab20")
    ax.scatter(
        x=group["frame_in_video"],
        y=d_burrow,
        c=group["traj_clip_id"],
        s=2.5,
        cmap=cmap,
    )
    ax.set_xlabel("frame in video")
    ax.set_ylabel("$d_{burrow}$ (pixels)")
    ax.set_title(
        f"Video: {video_str} ({n_frames / ds_video.fps / 60:.1f} min); burrow ID{b_id}"
    )

    # right: trajectories in burrow coord syst, coloured by frame number
    sc = ax_traj.scatter(
        x=group["x_burrow"],
        y=group["y_burrow"],
        c=group["frame_in_video"],
        s=2.5,
        cmap="viridis",
    )
    ax_traj.scatter(x=0, y=0, s=30, marker="x", color="r", zorder=5)
    fig.colorbar(sc, ax=ax_traj, label="frame in video")
    ax_traj.set_aspect("equal")
    ax_traj.invert_yaxis()  # match image coordinates (y down)
    ax_traj.set_xlabel("$x_{burrow}$ (pixels)")
    ax_traj.set_ylabel("$y_{burrow}$ (pixels)")
    ax_traj.set_title(f"burrow ID {b_id}")
# %%
