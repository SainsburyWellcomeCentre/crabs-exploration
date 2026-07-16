# %%
from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import xarray as xr
import zarr
from matplotlib.lines import Line2D
from mpl_toolkits.axes_grid1.anchored_artists import AnchoredSizeBar
from scipy.ndimage import center_of_mass

# %%
# %matplotlib qt
# qt / widget

# %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
# Parameters

# definition of visited burrows
min_samples_in_burrow_frac = 0.10

# minimum prominence (in body lengths) for a d_burrow excursion to count as a
# local peak, i.e. how far the crab must move away from the burrow and back
# min_peak_prominence_bl = 1.0

# localising peaks
# min drop in d_burrow_bl between consecutive frames
min_peak_drop_bl = 0.05
min_seconds_to_prev_peak = 2
min_d_burrow_at_peak_bl = 1.0

# localising inter-peak minima: after a peak, the first sample where the
# distance to burrow does not decrease AND is already below this value (BL)
max_d_burrow_at_min_bl = 0.25

# phase colours: inbound red, outbound taupe gray, and (in the phase plots) the
# samples that fall in no leg drawn in a distinct cool tone. The peak / min
# markers reuse the inbound / outbound colours.
color_inbound = "tab:red"
color_outbound = "#b6e2c8ff"
color_outbound_traj_plot = "#c1ddcd"
color_unassigned = "#7FA0B8"  # soft slate blue


# %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
# Helper functions
def factorize_string_coord_in_da(da, coord_name):
    coord_da = da[coord_name]
    coords_factorized, coords_unique = pd.factorize(coord_da.values)
    return coord_da.copy(
        data=coords_factorized
    ), coords_unique  # same dims, int data


# TODO: review!
# def flat_valid(input_da, ref_da, valid):
#     # align the 1-D coord to x_da's dims, broadcast as a VIEW, then mask
#     input_da_broadcasted = input_da.broadcast_like(ref_da)  # keeps it lazy/strided
#     return np.broadcast_to(input_da_broadcasted.values, ref_da.shape).reshape(-1)[valid]


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

output_figs_dir = Path(
    "/Users/sofia/arc/project_Zoo_crabs/ICN poster/figures/Fig-case-study-1"
)


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

# bbox size per sample: diagonal of the bbox sqrt(w**2 + h**2).
# The diagonal ~ body length and is stable under in-plane rotation (an
# axis-aligned bbox reshapes w/h as the crab turns, but the diagonal stays
# ~constant), unlike the geometric mean sqrt(w * h) which tracks area and
# dips toward 0 near axis alignment.
# (same dims as position, so the flattened order and `valid` mask match)
shape_da = ds_video.shape
bbox_w = shape_da.sel(space="x").values.reshape(-1)[valid]
bbox_h = shape_da.sel(space="y").values.reshape(-1)[valid]
bbox_diag = np.hypot(bbox_w, bbox_h)

# %%
# Free from memory
del position_da, x, y
del shape_da, bbox_w, bbox_h


# %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
# Compute traj-clip ids

indiv_code_da, unique_indivs = factorize_string_coord_in_da(
    x_da, "individuals"
)
clip_code_da, uniq_clip_ids = factorize_string_coord_in_da(x_da, "clip_id")

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
# Compute frame index per sample (wrt clip and video)
# 0-based indexing

# wrt clip
frame_in_clip_per_sample = (
    x_da["time"].broadcast_like(x_da).values.reshape(-1)[valid]
)

# wrt video
frame_in_video_per_sample = (
    frame_in_clip_per_sample
    + x_da["clip_first_frame_0idx"]
    .broadcast_like(x_da)
    .values.reshape(-1)[valid]
)

# determine if frame in escape period
frame_in_escape_period = ds_video.escape_state.broadcast_like(
    x_da
).values.reshape(-1)[valid]


# %%
# Free from memory
del valid, unique_indivs, uniq_clip_ids
del indiv_id_as_int, clip_id_as_int, combined

# %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
# Compute subset of visited burrows
# TODO: to be changed when I move to zarr with video-first indexing
list_videos = [lv.name for lv in dt.leaves]
video_idx_zarr = np.argwhere(video_str == np.asarray(list_videos)).item()
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
# IMPORTANT:
# - each trajectory is linked to only one burrow
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
        "frame_in_escape": frame_in_escape_period,
        "bbox_diag": bbox_diag,
    }
)

# polar rho coordinate in BCS in pixels
df["d_burrow_px"] = np.hypot(
    df["x_burrow"],
    df["y_burrow"],
)

# polar theta coordinate in BCS
df["theta_burrow"] = np.arctan2(
    df["y_burrow"],
    df["x_burrow"],
)

# %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
# Define dataframe for trajectories linked to a visited burrow only
# (used by most plots)
df_linked = df.dropna(subset=["burrow_id"]).copy()


# %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
# Define filtered dataframe,
# resolving clashes for trajectories linked to the same burrow
# (i.e. when 2 or more trajectories are present at the same frame)
# by keeping only the longest trajectory
#
# If two or more trajectory-clip IDs linked to the same burrow exist at the
# same frame, only one can be kept. We drop the loser(s) *entirely* (all their
# samples), not just the clashing rows.

# compute traj IDs sorted by number of sampels
traj_ids_by_size = (
    df_linked.groupby("traj_clip_id").size().sort_values(ascending=False).index
)

# A "slot" is a single (burrow_id, frame_in_video) pair.
# Greedy resolution: process trajectories
# from largest to smallest and keep one only if none of its (burrow, frame)
# slots are already taken by an already-kept trajectory. This guarantees the
# kept set is conflict-free for any number of overlapping trajectories.
burrow_id_frame_per_traj = (
    df_linked.groupby("traj_clip_id")
    .apply(
        lambda g: set(zip(g["burrow_id"], g["frame_in_video"], strict=True)),
        include_groups=False,
    )
    .loc[traj_ids_by_size]
)

occupied: set = set()
traj_ids_to_keep = []
for traj_id, burrow_id_frame in burrow_id_frame_per_traj.items():
    if burrow_id_frame.isdisjoint(occupied):
        traj_ids_to_keep.append(traj_id)
        # apply union update to the set
        occupied |= burrow_id_frame

df_linked_filtered = df_linked[
    df_linked["traj_clip_id"].isin(traj_ids_to_keep)
].copy()

# %%
# Free from memory
# (per-sample arrays now live in the dataframe)
del x_traj, y_traj, traj_clip_id_dense_per_sample
del frame_in_clip_per_sample, frame_in_video_per_sample, bbox_diag
del burrow_id_per_sample, centroid_xy_per_sample
del burrow_id_per_traj, centroid_xy_per_traj
del df_linked  # ok?


# %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
# Compute normalised position in burrow coord system (BCS),
# in cartesian and polar coordinates. Normalised by body
# lengths.

# compute normalising factor collecting data per burrow_id
# (i.e., all trajectories linked to a burrow). This derives
# one normalising factor per burrow, and implicitly assumes
# that all tracklets linked to a burrow belong to similarly
# sized crabs.
median_bbox_diag_per_burrow_id = df_linked_filtered.groupby("burrow_id")[
    "bbox_diag"
].transform("median")  # median per BURROW ID

# cartesian normalised position in BCS
df_linked_filtered["x_burrow_bl"] = (
    df_linked_filtered["x_burrow"] / median_bbox_diag_per_burrow_id
)
df_linked_filtered["y_burrow_bl"] = (
    df_linked_filtered["y_burrow"] / median_bbox_diag_per_burrow_id
)

# polar normalised rho coordinate in BCS
df_linked_filtered["d_burrow_bl"] = np.hypot(
    df_linked_filtered["x_burrow_bl"],
    df_linked_filtered["y_burrow_bl"],
)
# %%
fps = float(ds_video.fps)


# %%%%%%%%%%%%%%%%%%%%%%%
# Compute local peaks and inter-peak minima in d_burrow_bl
# Mark each sample as a peak in distance to burrow or an
# inter-peak minimum (closest approach between two peaks) with boolean columns.
#
# - Peaks are found per trajectory-clip on time-sorted samples, so we never
#   join across the time gaps between distinct trajectories. The prominence
#   threshold is in body lengths, so it adapts to crabs imaged at different
#   scales.
# - Minima are pooled per burrow: in each gap between consecutive peaks we pick
#   the single sample closest to the burrow, regardless of trajectory ID.
#
# NOTE: within a burrow, frame_in_video is unique (the clash-resolution filter
# guarantees no two kept trajectories share a (burrow, frame) slot), so each
# peak/min can be marked unambiguously on its own sample row.
df_linked_filtered["is_peak"] = (
    False  # all peaks, also those not followed by min
)
df_linked_filtered["is_min"] = False

n_frames_diff_wrt_prev_peak = fps * min_seconds_to_prev_peak

for _, df_trajs_b_id in df_linked_filtered.groupby("burrow_id"):
    # peaks per trajectory-clip: a peak is the first frame of a sharp,
    # consecutive-frame drop in distance to burrow (>= min_peak_drop_bl in a
    # single frame step). Frame gaps (missing detections) are never bridged,
    # and a multi-frame steep descent (several qualifying steps in a row)
    # only yields one peak, at its first frame.
    peak_index = []
    for _, traj in df_trajs_b_id.groupby("traj_clip_id"):
        traj = traj.sort_values("frame_in_video")
        frames_arr = traj["frame_in_video"].to_numpy()
        d_arr = traj["d_burrow_bl"].to_numpy()

        is_consec = np.diff(frames_arr) == 1
        is_drop = np.diff(d_arr) <= -min_peak_drop_bl
        is_far = (
            d_arr[:-1] >= min_d_burrow_at_peak_bl
        )  # height at start of drop
        candidate_peak = is_consec & is_drop & is_far

        # frame number for each candidate_peak
        frame_at_i = frames_arr[1:]
        last_peak_frame = np.maximum.accumulate(
            np.where(candidate_peak, frame_at_i, -1)
        )
        prev_last_peak_frame = np.r_[-1, last_peak_frame[:-1]]
        first_of_run = candidate_peak & (
            frame_at_i - prev_last_peak_frame > n_frames_diff_wrt_prev_peak
        )

        peak_idx = np.argwhere(first_of_run).reshape(-1)
        peak_index.extend(traj.index[peak_idx])
    df_linked_filtered.loc[peak_index, "is_peak"] = True

    # inter-peak minima: for each peak, the first sample afterwards where the
    # distance to burrow does not decrease (diff >= 0 across a
    # consecutive-frame step) AND is below max_d_burrow_at_min_bl. Pooled
    # across trajectories in the burrow; frame gaps are never bridged.
    group_sorted = df_trajs_b_id.sort_values("frame_in_video")
    frames_all = group_sorted["frame_in_video"].to_numpy()
    d_arr_all = group_sorted["d_burrow_bl"].to_numpy()
    index_arr = group_sorted.index.to_numpy()

    is_consec = np.diff(frames_all) == 1
    not_decreasing = np.diff(d_arr_all) >= 0  # [j] -> step j -> j+1
    # distance at the candidate min sample (sample j of step j -> j+1)
    # TODO: condition could also be: inside burrow?
    close_to_burrow = d_arr_all[:-1] < max_d_burrow_at_min_bl
    candidate_min = is_consec & not_decreasing & close_to_burrow

    peak_frames = np.sort(
        df_linked_filtered.loc[peak_index, "frame_in_video"].to_numpy()
    )
    min_index = []
    for f_peak in peak_frames:
        peak_pos = np.searchsorted(frames_all, f_peak)
        idcs_after_peak = np.argwhere(candidate_min[peak_pos:])
        if idcs_after_peak.size:
            # we take one sample before because at j+1 the distance has
            # started to grow
            min_index.append(index_arr[peak_pos + idcs_after_peak[0].item()])
    df_linked_filtered.loc[min_index, "is_min"] = True


# %%%%%%%%%%%%%%%%%%%%%%%
# Compute dataframe of inbound/outbound trajectories and metrics
# Build one row per inbound or outbound leg, holding its rho_dot (rate of
# change of distance to burrow) and its path tortuosity. Inbound legs are
# peak->min pairs; each one is followed by exactly one outbound leg, running
# from its min either to the next inbound leg's peak or (for the last one) to
# the recording edge. So there are always as many outbound legs as inbound
# legs, they tile the time from the first peak to the end of the recording, and
# unpaired peaks/mins are absorbed into the outbound legs. Only the span before
# the first peak is left unassigned.
#
# CONVENTION: legs OWN a non-overlapping, closed window of frames
# [frame_start, frame_end] (both ends included). frame_start is the event that
# starts the leg (peak for inbound, min for outbound). frame_end is the frame
# just before the next leg starts: min - 1 for inbound, next peak - 1 for
# outbound, and the last recorded frame for the final outbound leg (which has
# no next leg). So inbound runs from its peak to just before its min, outbound
# from its min to just before the next peak, and every sample belongs to at
# most one leg. Anything that assigns samples to legs (the phase colouring and
# the leg plots below) must use these bounds.
#
# The METRICS below deliberately use a wider window than the leg owns: they
# also read the closing event sample (the next leg's frame_start), i.e. the
# closed window [frame_start, frame_end + 1]. This is because both metrics are
# defined over *steps between samples*, not over samples:
# - rho_dot needs the step into the closing event (for inbound, the final
#   approach step into the min). Note the steps still tile exactly and are
#   never double counted: the inbound leg's last step is (min - 1 -> min) and
#   the following outbound leg's first step is (min -> min + 1).
# - tortuosity needs the closing event as its endpoint, so that the inbound
#   beeline is the true peak <-> min displacement.
# The final outbound leg has no closing event, so its metric window just ends
# at the last recorded frame.
#
# This dataframe construction relies on frame uniqueness within a burrow.
#
# - rho_dot: mean per-frame change in distance to burrow (body lengths/frame),
#   using only consecutive-frame steps; frame gaps (incl. the gaps between
#   has no consecutive-frame steps.
# - tortuosity: path length / beeline over the whole leg (dimensionless ratio).
#   The beeline is the straight peak<->min distance, so the path bridges any
#   gaps to span the same endpoints (NOTE: a large gap adds one long straight
#   segment). NaN when < 2 samples or zero beeline.
#
# Both metrics read the body-length burrow columns.

leg_records = []
for b_id, df_trajs_b_id in df_linked_filtered.groupby("burrow_id"):
    group_sorted = df_trajs_b_id.sort_values("frame_in_video")
    frames = group_sorted["frame_in_video"].to_numpy()
    d_bl = group_sorted["d_burrow_bl"].to_numpy()
    xs_all = group_sorted["x_burrow_bl"].to_numpy()
    ys_all = group_sorted["y_burrow_bl"].to_numpy()

    # time-ordered peak/min events
    event_mask = (group_sorted["is_peak"] | group_sorted["is_min"]).to_numpy()
    event_frames = frames[event_mask]  # start and end frames
    event_is_peak = group_sorted["is_peak"].to_numpy()[event_mask]

    # inbound legs: each adjacent peak->min in the event stream. A peak not
    # followed by a min never starts an inbound leg (it is skipped).
    inbound_spans = [
        (event_frames[s], event_frames[s + 1])
        for s in range(event_frames.size - 1)
        if event_is_peak[s] and not event_is_peak[s + 1]
    ]

    # outbound legs: the time between consecutive inbound legs, from the min
    # ending one inbound leg to the peak starting the next. Any skipped
    # peaks/mins in between are absorbed into a single outbound leg. The span
    # after the last inbound leg is also outbound, bounded
    # by the recording edge rather than by an event. A burrow with no inbound
    # leg gets no legs at all (rather than one whole-recording outbound leg).
    outbound_spans = [
        (inbound_spans[i][1], inbound_spans[i + 1][0])
        for i in range(len(inbound_spans) - 1)
    ]
    if inbound_spans:
        outbound_spans = (
            outbound_spans
            # post last inbound: no next leg, so stand in one past the last
            # recorded frame; this makes the leg own up to the last recorded
            # frame and its metric window stop there too
            + [(inbound_spans[-1][1], frames[-1] + 1)]
        )

    for kind, spans in [
        ("inbound", inbound_spans),
        ("outbound", outbound_spans),
    ]:
        for f0, next_leg_start in spans:
            # spans hold (own start event, next leg's start event); the leg
            # OWNS the closed frame window [f0, f1], ending just before the
            # next leg starts (zero overlap between legs)
            f1 = next_leg_start - 1

            # the METRICS window extends one event further, up to and
            # including the closing event, so that step-based metrics keep the
            # step into it (see the convention note above). For the final
            # outbound leg next_leg_start is one past the last recorded frame,
            # so this adds nothing.
            mask_frames_in_leg = (frames >= f0) & (frames <= next_leg_start)
            frames_leg = frames[mask_frames_in_leg]
            d_leg = d_bl[mask_frames_in_leg]
            xs, ys = xs_all[mask_frames_in_leg], ys_all[mask_frames_in_leg]

            # compute rho_dot_mean per leg: mean per-frame change in distance
            # to burrow, using only steps between consecutive frames
            # (step == 1); frame gaps are never bridged. NaN if the leg has no
            # consecutive-frame step.
            consecutive = np.diff(frames_leg) == 1  # per-step mask
            rho_dot = (
                np.diff(d_leg)[consecutive].mean()
                if consecutive.any()
                else np.nan
            )

            # tortuosity: path length/beeline over the whole leg, endpoints
            # included (for inbound: peak <-> min). The beeline is the straight
            # path between those endpoints, so the path length computation must
            # span the same endpoints. Gaps in the actual path are bridged by
            # line segments between known samples (a large gap adds one long
            # straight segment).
            if xs.size >= 2:
                path_len = np.hypot(np.diff(xs), np.diff(ys)).sum()
                beeline_xy = ((xs[0], ys[0]), (xs[-1], ys[-1]))
                beeline = np.hypot(
                    beeline_xy[1][0] - beeline_xy[0][0],
                    beeline_xy[1][1] - beeline_xy[0][1],
                )
                tort = path_len / beeline if beeline != 0 else np.nan
            else:
                beeline_xy = ((np.nan, np.nan), (np.nan, np.nan))
                tort = np.nan

            leg_records.append(
                {
                    "burrow_id": b_id,
                    "kind": kind,
                    "frame_start": f0,
                    "frame_end": f1,
                    "rho_dot_bl_mean": rho_dot,
                    "tortuosity": tort,
                    # the beeline endpoints that tortuosity divides by, kept so
                    # plots can draw the very same segment instead of
                    # re-deriving it (they are samples of the metric window, so
                    # the closing one can sit one frame past frame_end)
                    "beeline_x_start": beeline_xy[0][0],
                    "beeline_y_start": beeline_xy[0][1],
                    "beeline_x_end": beeline_xy[1][0],
                    "beeline_y_end": beeline_xy[1][1],
                }
            )

legs_df = pd.DataFrame(leg_records)

# %%%%%%%%%%%%%%%%%%%%%%%
# PLOTS
###%%%%%%%%%%%%%%%%%%%%%
# %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
# Plot selection method for visited burrows

fig, ax = plt.subplots(1, 1)

# plot hits per burrow id
ax.bar(
    np.arange(hits_per_id.shape[0])[1:],
    hits_per_id[1:],
)

# plot hline for min hits per burrow
ax.hlines(
    y=min_hits_per_burrow,
    xmin=0,
    xmax=hits_per_id.shape[0] + 1,
    colors="r",
)
ax.set_xlim([1, hits_per_id.shape[0] - 1])  # ok?
ax.set_xlabel("burrow ID")
ax.set_ylabel("trajectory samples")
ax.set_title(
    f"n = {sum(hits_per_id[1:] >= min_hits_per_burrow)} selected burrows"
)

# %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
# Plot visited burrow centroids on image plane
# (one centroid per visited burrow, read from the dataframe)

# dataframe of centroids indexed by burrow_id
visited_burrow_centroids = df_linked_filtered.groupby("burrow_id")[
    ["burrow_centroid_x", "burrow_centroid_y"]
].first()

fig, ax = plt.subplots(1, 1)

# plot masks of selected burrows
ax.imshow(
    np.isin(
        burrow_id_mask,
        visited_burrow_centroids.index.to_numpy("int64"),
    ),
)

# plot centroids
for burrow_id, (cx, cy) in visited_burrow_centroids.iterrows():
    ax.scatter(
        x=cx,
        y=cy,
        s=15,
        marker="x",
    )
    ax.text(x=cx, y=cy, s=str(burrow_id), color="w")


# %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
# Plot burrows and their linked trajectories
# POSTER FIGURE 1

fig, ax = plt.subplots(1, 1)

# background image (filename matches the video name)
background_img = plt.imread(raster_plots_dir / f"{video_str}.png")
# The PNG is a single flat colour (yellow-green) with a binary alpha mask —
# no intensity encoded. Recolour by keeping the mask and painting it blue.
alpha = background_img[..., 3]
recolored = np.zeros_like(background_img)
recolored[..., :3] = mpl.colors.to_rgb("0.8")  # light gray (0=black, 1=white)
recolored[..., 3] = alpha
ax.imshow(recolored)

cmap = plt.get_cmap("tab20")
for k, (burrow_id, df_trajs_b_id) in enumerate(
    df_linked_filtered.groupby("burrow_id")
):
    color = cmap(k % cmap.N)
    ax.scatter(
        x=df_trajs_b_id["x"],
        y=df_trajs_b_id["y"],
        s=0.5,
        marker=".",
        edgecolors=None,
        color=color,
        rasterized=True,
    )
    # burrow contour
    # ax.contour(
    #     burrow_id_mask == burrow_id,
    #     levels=[0.5],
    #     colors=[color],
    #     linewidths=1,
    # )

ax.set_title(f"{video_str} ({n_frames / fps / 60:.1f} min)")
ax.set_aspect("equal")
ax.set_xticks([])
ax.set_yticks([])
ax.margins(0)  # drop the 5% data padding around the artists
fig.subplots_adjust(left=0, right=1, bottom=0, top=1)  # kill figure padding

# %%
# ax.set_title(f"Sample video ({n_frames / fps / 60:.1f} min)")
# fig.savefig(
#     output_figs_dir / f"{video_str}_fig1_3.png",
#     dpi=300,  # resolution of the rasterized scatter/image
#     bbox_inches="tight",
#     pad_inches=0,  # no border around the tight bbox
# )

# %%%%%%%%%%%%%%%%%%%%%%%%%%
# Set text in axes as editable
plt.rcParams["svg.fonttype"] = "none"

# %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
# Plot all trajectories in burrow coordinate system
# and in units of bodylengths (BL)
# POSTER FIGURE 2

# Here, normalise by median BL of all trajectories in the plot
# rather than one factor per burrw

median_BL_all_visited_burrows = df_linked_filtered["bbox_diag"].median()

fig, ax = plt.subplots(1, 1, figsize=(10, 10))
cmap = plt.get_cmap("tab20")
ax.scatter(
    x=df_linked_filtered["x_burrow"] / median_BL_all_visited_burrows,
    y=df_linked_filtered["y_burrow"] / median_BL_all_visited_burrows,
    s=0.5,
    color=cmap(0),
    alpha=0.05,
    rasterized=True,
)

ax.scatter(x=0, y=0, s=30, marker="x", color="k", zorder=5)

# x-axis (red) and y-axis (green) unit vectors of the burrow coord system
axis_len = 2.0  # BL
ax.quiver(
    [0, 0],
    [0, 0],
    [axis_len, 0],
    [0, axis_len],
    color=[[1, 0, 0], [0, 1, 0]],
    angles="xy",
    scale_units="xy",
    scale=1,
    width=0.006,
    zorder=6,
)

# rings at radial percentiles: each circle encloses a given fraction of
# detections, so closely-spaced rings indicate high density. Each ring gets a
# distinct colour and is identified through the legend rather than an inline
# label.
ring_percentiles = [50, 75, 95, 100]
ring_radii = np.percentile(
    df_linked_filtered["d_burrow_px"] / median_BL_all_visited_burrows,
    ring_percentiles,
)
ring_cmap = plt.get_cmap("viridis")
ring_handles = []
for k, (pct, radius) in enumerate(
    zip(ring_percentiles, ring_radii, strict=True)
):
    color = ring_cmap(k / (len(ring_percentiles) - 1))
    ax.add_patch(
        plt.Circle(
            (0, 0),
            radius,
            fill=False,
            edgecolor=color,
            linewidth=2.5,
            zorder=4,
        )
    )
    ring_handles.append(
        mpl.lines.Line2D(
            [],
            [],
            color=color,
            linewidth=1.5,
            label=f"{pct}% ({radius:.1f} BL)",
        )
    )

ax.legend(
    handles=ring_handles,
    loc="upper right",
    fontsize=16,
    # title="radial percentile",
)

# scale bar spanning one body length. The axes are already in BL, so the bar
# has length 1 in data units. The caption gives the exact px conversion (a
# measured value) and the approximate physical size (~ / range, since the crab
# body length is only estimated to 5-7 cm).
scalebar = AnchoredSizeBar(
    ax.transData,
    1,  # bar length in data units (= 1 BL)
    f"1 BL = {median_BL_all_visited_burrows:.1f} px ≈ 5-7 cm",
    loc="lower left",
    pad=0.5,
    color="k",
    frameon=False,
    size_vertical=0.15,
    sep=4,  # # gap (in points) between bar and caption
    fontproperties=mpl.font_manager.FontProperties(size=16),
)
scalebar._box.align = "left"  # or "right"; default "center"
ax.add_artist(scalebar)

ax.set_aspect("equal")
ax.invert_yaxis()  # match image coordinates (y down)

for item in [ax.xaxis.label, ax.yaxis.label]:
    item.set_fontsize(16)
ax.tick_params(axis="both", labelsize=14)

ax.set_xlabel("$x_{burrow}$ (BL)")
ax.set_ylabel("$y_{burrow}$ (BL)")
# ax.set_title("Trajectories in burrow coord syst")

# %%
fig.savefig(
    output_figs_dir / f"{video_str}_polar_coords.svg",
    dpi=300,  # resolution of the rasterized scatter layer only
    bbox_inches="tight",
    pad_inches=0,  # no border around the tight bbox
)

# %%%%%%%%%%%%%%%%%%%%%%%%%%
# Polar histogram of angles relative to x-axis in BCS
n_bins = 36
counts, bin_edges = np.histogram(
    df_linked_filtered["theta_burrow"], bins=n_bins, range=(-np.pi, np.pi)
)
bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2
bin_width = bin_edges[1] - bin_edges[0]

# -------------
# polar histogram
fig, ax = plt.subplots(subplot_kw={"projection": "polar"})
ax.bar(bin_centers, counts, width=bin_width, bottom=0.0, align="center")

ax.set_theta_zero_location("E")
ax.set_theta_direction(-1)  # theta increases in clockwise direction
ax.set_rlabel_position(270)  # put count labels in an empty-ish quadrant
# ax.set_title("Angle of detections relative to burrow centroid")

# drop the concentric constant-r gridlines (and their labels); the radial
# scale isn't the message of this plot
ax.yaxis.grid(False)
ax.set_yticklabels([])

# -----------------------
# compute circular mean direction (mean resultant vector).
# theta is circular, so the mean direction is the angle of the resultant vector
# z = <e^{i*theta}>, NOT np.mean(theta) (which depends on the +/-pi branch cut
# and breaks across the wrap).
theta = df_linked_filtered["theta_burrow"].to_numpy()
z = np.mean(np.exp(1j * theta))
mu = np.angle(z)

# mean-direction ray on top of the fans
ax.plot(
    [mu, mu],
    [0, counts.max()],
    color="purple",
    linewidth=2.5,
    linestyle="--",
    zorder=6,
    label=rf"$\mu = {np.degrees(mu):.0f}$°",
)

# -------------------
# symmetric angle fans centred on the mean direction, each containing an
# increasing fraction of detections (the circular analogue of mu +/- k*sigma).
# The half-width for a given coverage is read empirically off the absolute
# angular deviation from mu, so it is distribution-free and rotation-invariant
# (no branch-cut caveat). NOTE: this assumes a unimodal, roughly symmetric
# distribution about mu; if the angles are bimodal the fans mislead.
# absolute angular deviation from mu, wrapped into [0, pi]
dev = np.abs((theta - mu + np.pi) % (2 * np.pi) - np.pi)
fan_coverages = [50, 75, 95]  # % of detections inside each fan
fan_half_widths = np.percentile(dev, fan_coverages)  # radians

# draw each fan as just its two lateral (radial) edges at mu +/- hw, leaving
# the wedge itself transparent. Each fan's arc sits at a staggered radius so the
# nested fans don't overlap and can be told apart.
fan_cmap = plt.get_cmap("viridis")
# radii evenly spaced from x% to 100% of the count axis, one per fan
fan_radii = np.linspace(0.97, 1.0, len(fan_coverages)) * counts.max()
for k, (pct, hw, fan_len) in enumerate(
    zip(fan_coverages, fan_half_widths, fan_radii, strict=True)
):
    # match the fig-2 ring colours: those normalise over [50,75,95,100], so
    # divide by len(fan_coverages) (not len-1) to land 50/75/95 on the same
    # viridis positions (0, 1/3, 2/3) without introducing a 100% fan
    color = fan_cmap(k / len(fan_coverages))
    # label only one edge so the legend has a single entry per fan
    ax.plot(
        [mu - hw, mu - hw],
        [0, fan_len],
        color=color,
        linewidth=2.5,
        zorder=5,
        label=f"{pct}% (±{np.degrees(hw):.0f}°)",
    )
    ax.plot(
        [mu + hw, mu + hw],
        [0, fan_len],
        color=color,
        linewidth=2.5,
        zorder=5,
    )
    # arc joining the two edges at this fan's (staggered) outer radius
    arc_theta = np.linspace(mu - hw, mu + hw, 100)
    ax.plot(
        arc_theta,
        np.full_like(arc_theta, fan_len),
        color=color,
        linewidth=2.5,
        zorder=5,
    )


# legend
ax.legend(
    loc="upper right",
    bbox_to_anchor=(1.15, 1.1),
    fontsize=12,
)


# x-axis (red) and y-axis (green) axes of the burrow coord system.
# On a polar axes, arrows are drawn with annotate in (theta, r) data coords:
# theta=0 points +x (East) and theta=pi/2 points +y (down, given the clockwise
# direction and East zero location set above).
# axis_len = 40_000  # counts
# for theta, color in [(0, [1, 0, 0]), (np.pi / 2, [0, 1, 0])]:
#     ax.annotate(
#         "",
#         xy=(theta, axis_len),
#         xytext=(0, 0),
#         arrowprops=dict(color=color, arrowstyle="->", linewidth=2),
#         zorder=6,
#     )


# %%
fig.savefig(
    output_figs_dir / f"{video_str}_theta_histogram.svg",
    bbox_inches="tight",
    pad_inches=0,  # no border around the tight bbox
)


# %%%%%%%%%%%%%%%%%%%%%%%
# Plot single burrow metrics poster


# selected burrow
b_id = 53  # 21

# get trajectories for one burrow
df_trajs_b_id = df_linked_filtered[df_linked_filtered["burrow_id"] == b_id]

legs_one_burrow = legs_df[legs_df["burrow_id"] == b_id]
inbound_legs_df = legs_one_burrow[legs_one_burrow["kind"] == "inbound"]
outbound_legs_df = legs_one_burrow[legs_one_burrow["kind"] == "outbound"]

# get the peak/min event samples: each leg's frame_start is its starting
# event (peak for inbound, min for outbound); an inbound leg's frame_end is
# the min - 1, NOT the min itself (closed-window convention).
# (Frames are unique within a burrow, so we can select those samples by frame)
peaks_df = df_trajs_b_id[
    df_trajs_b_id["frame_in_video"].isin(inbound_legs_df["frame_start"])
]
mins_df = df_trajs_b_id[
    df_trajs_b_id["frame_in_video"].isin(outbound_legs_df["frame_start"])
]

# %%%%%%%%%%%
# plot trajectories in burrow coord syst, coloured by frame number
fig, ax_traj = plt.subplots(figsize=(10, 10))
sc = ax_traj.scatter(
    x=df_trajs_b_id["x_burrow_bl"],
    y=df_trajs_b_id["y_burrow_bl"],
    c=df_trajs_b_id["frame_in_video"] / fps / 60,
    s=2.5,
    cmap="viridis",
    rasterized=True,
)
ax_traj.scatter(x=0, y=0, s=30, marker="x", color="r", zorder=5)
cbar = fig.colorbar(sc, ax=ax_traj, location="left")
cbar.set_label("time (min)", fontsize=18)
cbar.ax.tick_params(labelsize=16)

# scale bar spanning one body length. The axes are already in BL, so the bar
# has length 1 in data units. The caption gives the exact px conversion (a
# measured value) and the approximate physical size (~ / range, since the crab
# body length is only estimated to 5-7 cm).
scalebar = AnchoredSizeBar(
    ax_traj.transData,
    1,  # bar length in data units (= 1 BL)
    f"1 BL = {df_trajs_b_id['bbox_diag'].median():.1f} px ≈ 5-7 cm",  # BL for this plot
    loc="lower left",
    pad=0.5,
    color="k",
    frameon=False,
    size_vertical=0.15,
    sep=4,  # # gap (in points) between bar and caption
    fontproperties=mpl.font_manager.FontProperties(size=16),
)
scalebar._box.align = "left"  # default "center"
ax_traj.add_artist(scalebar)

ax_traj.set_aspect("equal")
ax_traj.invert_yaxis()  # match image coordinates (y down)
ax_traj.set_xlabel("$x_{burrow}$ (BL)")
ax_traj.set_ylabel("$y_{burrow}$ (BL)")
ax_traj.set_title(f"burrow ID {b_id}")
ax_traj.axis("off")

# %%
fig.savefig(
    output_figs_dir / f"{video_str}_burrow_ID{b_id}_colbytime.svg",
    dpi=300,  # resolution of the rasterized scatter/image
    bbox_inches="tight",
    pad_inches=0,  # no border around the tight bbox
)


# %%%%%%%%%%%%%%%
# plot distance vs time
fig, ax = plt.subplots(figsize=(10, 10))
cmap = plt.get_cmap("tab20")
ax.scatter(
    x=df_trajs_b_id["frame_in_video"] / fps / 60,
    y=df_trajs_b_id["d_burrow_bl"],
    color=cmap(0),
    # c=cmap(0), #group["traj_clip_id"],
    s=2.5,
    # cmap=cmap,
    rasterized=True,
)
# mark detected local peaks
ax.scatter(
    x=peaks_df["frame_in_video"] / fps / 60,
    y=peaks_df["d_burrow_bl"],
    s=50,
    marker="v",
    facecolors=color_inbound,
    edgecolors=color_inbound,
    linewidths=2.5,
    zorder=6,
    label=f"peaks (n={len(peaks_df)})",
)
# mark detected local minima
ax.scatter(
    x=mins_df["frame_in_video"] / fps / 60,
    y=mins_df["d_burrow_bl"],
    s=50,
    marker="^",
    facecolors=color_outbound,
    edgecolors=color_outbound,  # matches outbound phase
    linewidths=2.5,
    zorder=6,
    label=f"inter-peak min (n={len(mins_df)})",
)
for item in [ax.xaxis.label, ax.yaxis.label]:
    item.set_fontsize(20)
ax.tick_params(axis="both", labelsize=18)


ax.legend(loc="upper right", fontsize=18)
ax.set_xlabel("time (min)")
ax.set_ylabel(r"$\rho$ (BL)")
# ax.set_title(
#     f"Video: {video_str} ({n_frames / fps / 60:.1f} min); burrow ID{b_id}"
# )
fig.tight_layout()

ax.spines[["top", "right"]].set_visible(False)

# %%
fig.savefig(
    output_figs_dir / f"{video_str}_burrow_ID{b_id}_rho_vs_time.svg",
    dpi=300,  # resolution of the rasterized scatter/image
    bbox_inches="tight",
    pad_inches=0,  # no border around the tight bbox
)

# %%%%%%%%%%%%%%%
# plot distance vs time, coloured by PHASE
# (inbound/outbound/unassigned)

# assign each sample the phase of the leg it falls in (colours from the
# parameters block): inbound / outbound, and samples outside every leg (i.e.
# before the first peak, the only unassigned span) in the unassigned colour.
# Legs are non-overlapping closed windows [frame_start, frame_end] (see the
# legs_df construction cell), so each sample gets exactly one phase.
phase_to_color = {"inbound": color_inbound, "outbound": color_outbound}
sample_color = pd.Series(
    color_unassigned, index=df_trajs_b_id.index
)  # samples in no leg
for _, leg in legs_one_burrow.iterrows():
    in_leg = df_trajs_b_id["frame_in_video"].between(
        leg["frame_start"],
        leg["frame_end"],  # inclusive on both ends
    )
    sample_color[in_leg] = phase_to_color[leg["kind"]]

# for this plot only: highlight the inbound legs in red and draw every other
# sample in the same base colour as the distance-vs-time plot above.
# (sample_color keeps the full phase colouring for the plots below)
color_base = mpl.colors.to_hex(plt.get_cmap("tab20")(0))
sample_color_inbound = sample_color.where(
    sample_color == color_inbound, color_base
)

fig, ax = plt.subplots(figsize=(10, 10))
ax.scatter(
    x=df_trajs_b_id["frame_in_video"] / fps / 60,
    y=df_trajs_b_id["d_burrow_bl"],
    c=sample_color_inbound,  # sample_color
    s=2.5,
    rasterized=True,
)
# mark detected local peaks
ax.scatter(
    x=peaks_df["frame_in_video"] / fps / 60,
    y=peaks_df["d_burrow_bl"],
    s=50,
    marker="v",
    facecolors=color_inbound,
    edgecolors=color_inbound,
    linewidths=2.5,
    zorder=6,
    label=f"peaks (n={len(peaks_df)})",
)
# mark detected local minima
ax.scatter(
    x=mins_df["frame_in_video"] / fps / 60,
    y=mins_df["d_burrow_bl"],
    s=50,
    marker="^",
    facecolors=color_outbound,
    edgecolors=color_outbound,  # matches outbound phase
    linewidths=2.5,
    zorder=6,
    label=f"inter-peak min (n={len(mins_df)})",
)
for item in [ax.xaxis.label, ax.yaxis.label]:
    item.set_fontsize(20)
ax.tick_params(axis="both", labelsize=18)


# legend: the inbound phase colour followed by the peak/min markers
# (phase_to_color_legend is kept for the trajectory-by-phase plot below)
phase_to_color_legend = phase_to_color.copy()
phase_to_color_legend.update({"unassigned": color_unassigned})
inbound_handle = Line2D(
    [], [], marker="o", linestyle="none", color=color_inbound, label="inbound"
)
peak_min_handles, _ = ax.get_legend_handles_labels()
ax.legend(
    handles=[inbound_handle] + peak_min_handles, loc="upper right", fontsize=18
)
ax.set_xlabel("time (min)")
ax.set_ylabel(r"$\rho$ (BL)")
# ax.set_title(
#     f"Video: {video_str} ({n_frames / fps / 60:.1f} min); burrow ID{b_id}"
# )
fig.tight_layout()

ax.spines[["top", "right"]].set_visible(False)

# %%
fig.savefig(
    output_figs_dir
    / f"{video_str}_burrow_ID{b_id}_rho_vs_time_outbound_red.svg",
    dpi=300,  # resolution of the rasterized scatter/image
    bbox_inches="tight",
    pad_inches=0,  # no border around the tight bbox
)


# %%%%%%%%%%%
# plot trajectories in burrow coord syst, coloured by PHASE
# (inbound / outbound / unassigned colours from the parameters block). Reuses
# sample_color and phase_to_color from the distance-vs-time phase cell above.

# for this plot only:
# - use a softer outbound tone and
# - hide unassigned samples
# (8-digit hex is RGBA; alpha 00 makes them fully transparent)
sample_color_plot = sample_color.replace(
    {
        color_outbound: color_outbound_traj_plot,
        color_unassigned: "#00000000",
    }
)


fig, ax_traj = plt.subplots(figsize=(10, 10))
ax_traj.scatter(
    x=df_trajs_b_id["x_burrow_bl"],
    y=df_trajs_b_id["y_burrow_bl"],
    c=sample_color_plot,
    s=2.5,
    rasterized=True,
)
ax_traj.scatter(x=0, y=0, s=30, marker="x", color="k", zorder=5)

# phase legend (categorical, so no colorbar);
# NOTE: unassigned samples are
# transparent in this plot, so they get no legend entry
phase_handles = [
    Line2D([], [], marker="o", linestyle="none", color=color, label=kind)
    for kind, color in phase_to_color_legend.items()
    if kind != "unassigned"
]
ax_traj.legend(handles=phase_handles, loc="upper right", fontsize=16)

# scale bar spanning one body length. The axes are already in BL, so the bar
# has length 1 in data units. The caption gives the exact px conversion (a
# measured value) and the approximate physical size (~ / range, since the crab
# body length is only estimated to 5-7 cm).
scalebar = AnchoredSizeBar(
    ax_traj.transData,
    1,  # bar length in data units (= 1 BL)
    f"1 BL = {df_trajs_b_id['bbox_diag'].median():.1f} px ≈ 5-7 cm",  # BL for this plot
    loc="lower left",
    pad=0.5,
    color="k",
    frameon=False,
    size_vertical=0.15,
    sep=4,  # # gap (in points) between bar and caption
    fontproperties=mpl.font_manager.FontProperties(size=16),
)
scalebar._box.align = "left"  # default "center"
ax_traj.add_artist(scalebar)

ax_traj.set_aspect("equal")
ax_traj.invert_yaxis()  # match image coordinates (y down)
ax_traj.set_xlabel("$x_{burrow}$ (BL)")
ax_traj.set_ylabel("$y_{burrow}$ (BL)")
ax_traj.set_title(f"burrow ID {b_id}")
ax_traj.axis("off")

# %%
fig.savefig(
    output_figs_dir
    / f"{video_str}_burrow_ID{b_id}_colbyphase_no_unassigned.svg",
    dpi=300,  # resolution of the rasterized scatter layer only
    bbox_inches="tight",
    pad_inches=0,  # no border around the tight bbox
)

# %%%%%%%%%%%%%%%
# plot angle to burrow (theta) at each peak, over time.
# theta is wrapped to [-45, 315) deg so the branch cut (the "switching"
# border) sits at -pi/4 rather than at +/-pi, keeping angular clusters
# from being split across the wrap.
peak_theta_deg = np.degrees(
    np.mod(peaks_df["theta_burrow"] + np.pi / 4, 2 * np.pi) - np.pi / 4
)

fig, ax_theta = plt.subplots(figsize=(8, 4.25))
ax_theta.plot(
    peaks_df["frame_in_video"] / fps / 60,
    peak_theta_deg,
    linestyle="--",
    color="r",
    alpha=0.4,
    linewidth=2,
)

ax_theta.scatter(
    x=peaks_df["frame_in_video"] / fps / 60,
    y=peak_theta_deg,
    s=40,
    marker="v",
    facecolors="none",
    edgecolors=color_inbound,
    linewidths=2,
    zorder=6,
    label=f"peaks (n={len(peaks_df)})",
)
ax_theta.set_yticks(np.arange(0, 316, 45))
# ax_theta.set_ylim(0, 315)  # (0, 315)
ax_theta.set_xlabel("time (min)")
ax_theta.set_ylabel(r"$\theta$ ($\degree$)")
ax_theta.spines[["top", "right"]].set_visible(False)

# ax_theta.set_title(f"Angle to burrow at peaks; burrow ID{b_id}")

for item in [ax_theta.xaxis.label, ax_theta.yaxis.label]:
    item.set_fontsize(18)
ax_theta.tick_params(axis="both", labelsize=16)
# %%
fig.savefig(
    output_figs_dir / f"{video_str}_burrow_ID{b_id}_theta_peak_vs_time.svg",
    # dpi=300,  # resolution of the rasterized scatter/image
    bbox_inches="tight",
    pad_inches=0,  # no border around the tight bbox
)

# %%
# plot speed of change of d_burrow on inbound vs outbound legs.
# take abs value and express in body lengths/s

labels = ["outbound\n(min → peak)", "inbound\n(peak → min)"]
colors = [color_outbound, color_inbound]
rng = np.random.default_rng(0)


fig, ax_rate = plt.subplots(figsize=(4, 4))
rho_dot_bl_per_s = [
    np.abs(outbound_legs_df["rho_dot_bl_mean"].dropna().to_numpy()) * fps,
    np.abs(inbound_legs_df["rho_dot_bl_mean"].dropna().to_numpy()) * fps,
]
for xpos, (r, color) in enumerate(zip(rho_dot_bl_per_s, colors, strict=True)):
    ax_rate.scatter(
        xpos + rng.uniform(-0.08, 0.08, size=r.size),  # jitter
        r,
        color=color,
        s=25,
        alpha=0.75,
        zorder=3,
    )
    # horizontal bar marking the mean of each group.
    if r.size:
        ax_rate.hlines(
            r.mean(),
            xpos - 0.25,
            xpos + 0.25,
            color="k",
            linewidth=2,
            zorder=4,
        )

ax_rate.set_xticks([0, 1])
ax_rate.set_xticklabels(labels)
ax_rate.set_xlim(-0.5, 1.5)
ax_rate.set_ylim(bottom=0)
ax_rate.set_ylabel(r"$|\overset{\bullet}{\rho}|$ (BL/s)")

fig.tight_layout()

for item in [ax_rate.xaxis.label, ax_rate.yaxis.label]:
    item.set_fontsize(18)
ax_rate.tick_params(axis="both", labelsize=16)

ax_rate.spines[["top", "right"]].set_visible(False)


# %%
fig.savefig(
    output_figs_dir / f"{video_str}_burrow_ID{b_id}_abs_rho_dot_vs_time.svg",
    # dpi=300,  # resolution of the rasterized scatter/image
    bbox_inches="tight",
    pad_inches=0,  # no border around the tight bbox
)

# %%
# fourth: path tortuosity (path length / beeline) on inbound vs outbound
# legs. Same strip-plot style as the rho_dot panel; tortuosity is >= 1, with
# 1 = perfectly straight.
fig, ax_tort = plt.subplots(figsize=(4, 4))
torts = [
    outbound_legs_df["tortuosity"].dropna().to_numpy(),
    inbound_legs_df["tortuosity"].dropna().to_numpy(),
]
for xpos, (t, color) in enumerate(zip(torts, colors, strict=True)):
    ax_tort.scatter(
        xpos + rng.uniform(-0.08, 0.08, size=t.size),  # jitter
        t,
        color=color,
        s=25,
        alpha=0.75,
        zorder=3,
    )
    if t.size:
        ax_tort.hlines(
            t.mean(),
            xpos - 0.25,
            xpos + 0.25,
            color="k",
            linewidth=2,
            zorder=4,
        )

# dashed line at 1 = perfectly straight path
ax_tort.axhline(1, color="k", linewidth=0.5, linestyle="--")
ax_tort.set_xticks([0, 1])
ax_tort.set_xticklabels(labels)
ax_tort.set_xlim(-0.5, 1.5)
ax_tort.set_yscale("log")
ax_tort.set_ylim(bottom=1)  # perfectly straight path
ax_tort.set_ylabel("tortuosity ratio")

fig.tight_layout()

for item in [ax_tort.xaxis.label, ax_tort.yaxis.label]:
    item.set_fontsize(18)
ax_tort.tick_params(axis="both", labelsize=16)
ax_tort.spines[["top", "right"]].set_visible(False)


mean_handle = Line2D([], [], color="k", linewidth=2, label="mean value")
n_handle = Line2D([], [], linestyle="none", label=f"n = {torts[0].size} legs")

ax_tort.legend(
    handles=[mean_handle, n_handle],
    loc="upper left",  # legend corner that gets anchored
    bbox_to_anchor=(1.02, 1.0),  # anchor point in axes fraction (>1 = outside)
    fontsize=14,
    borderaxespad=0,
)

# %%
fig.savefig(
    output_figs_dir / f"{video_str}_burrow_ID{b_id}_tortuosity_vs_time.svg",
    # dpi=300,  # resolution of the rasterized scatter/image
    bbox_inches="tight",
    pad_inches=0,  # no border around the tight bbox
)
# %%

# Plot a pair of consecutive outbound/inbound

plot_inbound_to_outbound = False

# prepare burrow contour
# The mask lives in image (pixel) space, but these axes are in the burrow
# coordinate system and in body lengths, so transform the pixel grid the same
# way the trajectory samples were: subtract the burrow centroid and divide by
# the burrow's body length. Crop to the burrow's bounding box first so we don't
# build a full-image meshgrid.
burrow_cx = df_trajs_b_id["burrow_centroid_x"].iloc[0]
burrow_cy = df_trajs_b_id["burrow_centroid_y"].iloc[0]
burrow_bl = df_trajs_b_id["bbox_diag"].median()

mask_b_id = burrow_id_mask == b_id
mask_rows, mask_cols = np.where(mask_b_id)
r0, r1 = mask_rows.min(), mask_rows.max()
c0, c1 = mask_cols.min(), mask_cols.max()
mask_b_id_crop = mask_b_id[r0 : r1 + 1, c0 : c1 + 1]

# pad the crop with a background border: contour cannot close a level line that
# runs into the edge of its data domain, and a tight bounding box has the mask
# touching all four edges (which leaves the contour open at the burrow's
# extremes). The padding surrounds the mask with zeros so the 0.5 level closes.
pad = 1
mask_b_id_crop = np.pad(mask_b_id_crop, pad)

contour_x_bl = (np.arange(c0 - pad, c1 + 1 + pad) - burrow_cx) / burrow_bl
contour_y_bl = (np.arange(r0 - pad, r1 + 1 + pad) - burrow_cy) / burrow_bl


for bound_idx in np.arange(inbound_legs_df.shape[0]):

    # plot an inbound+outbound pair
    if plot_inbound_to_outbound:
        # inbound leg i is followed by outbound leg i
        pair_legs = [
            inbound_legs_df.iloc[bound_idx],
            outbound_legs_df.iloc[bound_idx],
        ]
        title_str = "inbound -> outbound"

    # plot an outbound+inbound pair
    # (outbound leg i is followed by inbound leg i+1)
    else:
        pair_legs = [outbound_legs_df.iloc[bound_idx]]

        # the last outbound leg has no inbound leg after it (it runs to the
        # recording edge), so there the pair is that outbound leg alone
        if bound_idx + 1 < len(inbound_legs_df):
            pair_legs.append(inbound_legs_df.iloc[bound_idx + 1])

        title_str = "outbound -> inbound"

    # start = first leg's start event, end = last leg's frame_end
    start = pair_legs[0].frame_start
    end = pair_legs[-1].frame_end

    # legs are closed windows [frame_start, frame_end], so + 1 to include the
    # last frame of the second leg of the pair
    slc_traj = df_trajs_b_id.frame_in_video.isin(np.arange(start, end + 1))

    fig, axs = plt.subplots(1, 2, figsize=(10, 10))
    axs[0].scatter(
        x=df_trajs_b_id["x_burrow_bl"][slc_traj],
        y=df_trajs_b_id["y_burrow_bl"][slc_traj],
        c=sample_color_plot[slc_traj],
        s=10,
        rasterized=True,
    )

    # this one is just for reference so we rasterise
    sc = axs[1].scatter(
        x=df_trajs_b_id["x_burrow_bl"][slc_traj],
        y=df_trajs_b_id["y_burrow_bl"][slc_traj],
        c=df_trajs_b_id["frame_in_video"][slc_traj],
        s=2.5,
        rasterized=True,
    )

    for ax in axs:
        ax.contour(
            contour_x_bl,
            contour_y_bl,
            mask_b_id_crop,
            levels=[0.5],
            colors="k",
            linewidths=2.5,
        )

        # beelines: the straight segments tortuosity divides the path length
        # by, one per leg of the pair. They are drawn from the endpoints stored
        # when legs_df was built, so they are exactly the denominators of the
        # tortuosity values quoted in the legend. NOTE their closing endpoint
        # is the next leg's starting event (peak or min), which the leg does
        # not own, so a beeline can end one frame past the last sample plotted
        # here (visible for the inbound leg closing on its min).
        for leg in pair_legs:
            ax.plot(
                [leg.beeline_x_start, leg.beeline_x_end],
                [leg.beeline_y_start, leg.beeline_y_end],
                color=phase_to_color[leg.kind],
                linewidth=2.5,
                linestyle="--",
                marker="o",
                markersize=4,
                zorder=4,
                label=f"{leg.kind} beeline (tortuosity {leg.tortuosity:.2f})",
            )

        ax.scatter(x=0, y=0, s=30, marker="x", color="k", zorder=5)
        ax.set_aspect("equal")
        ax.invert_yaxis()  # match image coordinates (y down)
        ax.set_xlabel("$x_{burrow}$ (BL)")
        ax.set_ylabel("$y_{burrow}$ (BL)")
        ax.set_title(f"burrow ID {b_id}")
        # ax.axis("off")
        ax.set_title(f"Bound idx {bound_idx} - {title_str}")

    axs[0].legend(loc="best", fontsize=9)

    cbar = fig.colorbar(sc, ax=axs[1], location="right")

    # cbar.set_label("time (min)", fontsize=18)
    # cbar.ax.tick_params(labelsize=16)


    # # save figure
    # fig.savefig( 
    #     output_figs_dir /"Fig-single-outbound-inbound" / "rasterised" /f"{video_str}_burrow_ID{b_id}_{title_str}_idx{bound_idx}.svg",
    #     dpi=300,  # resolution of the rasterized scatter/image
    #     bbox_inches="tight",
    #     pad_inches=0,  # no border around the tight bbox
    # )
        

# %%
