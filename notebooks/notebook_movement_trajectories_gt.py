"""Inspect crab groundtruth clip using movement"""

# %%
import itertools

import matplotlib.pyplot as plt
import numpy as np
from cycler import cycler
from movement.io import load_bboxes

# %%%%%%%%%%
# Enable interactive plots
# %matplotlib widget

# %%%%%%%%%%%%%%%%%%%%%
# load predictions
file_csv = (
    "/Users/sofia/arc/project_Zoo_crabs/escape_clips/"
    "crabs_track_output_selected_clips/04.09.2023-04-Right_RE_test/predicted_tracks.csv"
)


# load ground truth (corrected)
groundtruth_csv_corrected = (
    "/Users/sofia/arc/project_Zoo_crabs/tracking_groundtruth_generation/"
    "04.09.2023-04-Right_RE_test_corrected_ST_SM_20241029_113207.csv"
)


# %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
# Read corrected ground truth as a movement dataset
ds_gt = load_bboxes.from_via_tracks_file(
    groundtruth_csv_corrected, fps=None, use_frame_numbers_from_file=False
)
print(ds_gt)

# Print summary
print(f"{ds_gt.source_file}")
print(f"Number of frames: {ds_gt.sizes['time']}")
print(f"Number of individuals: {ds_gt.sizes['individuals']}")

# %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
# Read predictions as a movement dataset
ds_pred = load_bboxes.from_via_tracks_file(
    file_csv, fps=None, use_frame_numbers_from_file=False
)
print(ds_pred)

# Print summary
print(f"{ds_pred.source_file}")
print(f"Number of frames: {ds_pred.sizes['time']}")
print(f"Number of individuals: {ds_pred.sizes['individuals']}")

# %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
# Check predicted and ground truth labels
# check x and y coordinates are nan at the same locations
# TODO: change colormap to white and blue
assert (
    np.isnan(ds_gt.position.data[:, :, 0])
    == np.isnan(ds_gt.position.data[:, :, 1])
).all()

fig, axs = plt.subplots(2, 1, sharex=True)

axs[0].matshow(np.isnan(ds_gt.position.data[:, :, 0]).T, aspect="auto")
axs[0].set_title("Ground truth")
axs[0].set_xlabel("time (frames)")
axs[0].set_ylabel("individual")

axs[1].matshow(np.isnan(ds_pred.position.data[:, :, 0]).T, aspect="auto")
axs[1].set_title("Prediction")
axs[1].set_xlabel("time (frames)")
axs[1].set_ylabel("tracks")
axs[1].xaxis.tick_bottom()


fig.subplots_adjust(hspace=0.6, wspace=0.5)

# %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
# Compare lengths of trajectories
non_nan_frames_gt = {}
for ind, id_str in enumerate(ds_gt.individuals):
    non_nan_frames_gt[ind] = (
        len(ds_gt.time)
        - ds_gt.position[:, ind, :].isnull().any(axis=1).sum().item()
    )

non_nan_frames_pred = {}
for ind, id_str in enumerate(ds_pred.individuals):
    non_nan_frames_pred[ind] = (
        len(ds_pred.time)
        - ds_pred.position[:, ind, :].isnull().any(axis=1).sum().item()
    )

# plot histogram
fig, ax = plt.subplots(1, 1)
ax.hist(
    non_nan_frames_gt.values(),
    bins=np.arange(0, len(ds_pred.time) + 50, 50),
    alpha=0.5,
    label="GT",
)
out = ax.hist(
    non_nan_frames_pred.values(),
    bins=np.arange(0, len(ds_pred.time) + 50, 50),
    alpha=0.5,
    label="Prediction",
)

ax.set_xlabel("n frames with same ID")
ax.set_ylabel("n tracks")
ax.tick_params(labeltop=False, labelright=True, right=True, which="both")
ax.hlines(y=len(ds_gt.individuals), xmin=0, xmax=len(ds_gt.time), color="red")
ax.legend(bbox_to_anchor=(1.0, 1.16))
# %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
# Check confidence of detections
confidence_values = ds_pred.confidence.data.flatten()
nan_median_confidence = np.nanmedian(confidence_values)


fig, ax = plt.subplots(1, 1)
hist = ax.hist(confidence_values, bins=np.arange(0, 1.01, 0.05))
ax.vlines(x=nan_median_confidence, ymin=0, ymax=max(hist[0]), color="red")
ax.set_aspect("auto")

fig, ax = plt.subplots(1, 1)
ax.hist(ds_pred.confidence.data.flatten(), bins=np.arange(0.6, 1.01, 0.01))
ax.vlines(x=nan_median_confidence, ymin=0, ymax=max(hist[0]), color="red")
ax.set_aspect("auto")

print(f"Median confidence: {nan_median_confidence}")

# %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
# plot GT and predicted trajectories
# ds.position ---> time, individuals, space
# why noise? remove low predictions?

flag_plot_ids = False

for ds, title in zip(
    [ds_gt, ds_pred], ["Ground truth", "Prediction"], strict=False
):
    fig, ax = plt.subplots(1, 1)
    plt.rcParams["axes.prop_cycle"] = cycler(
        color=plt.get_cmap("tab10").colors
    )
    color_cycle = plt.rcParams["axes.prop_cycle"].by_key()["color"]

    for ind_idx in range(ds.sizes["individuals"]):
        # plot trajectories
        ax.scatter(
            x=ds.position[:, ind_idx, 0],  # nframes, nindividuals, x
            y=ds.position[:, ind_idx, 1],
            s=1,
            # c=cmap(ind_idx),
        )
        # add ID at first frame with non-nan x-coord
        if flag_plot_ids:
            start_frame = ds.time[~ds.position.isnull()[:, ind_idx, 0]][
                0
            ].item()
            ax.text(
                x=ds.position[start_frame, ind_idx, 0],
                y=ds.position[start_frame, ind_idx, 1],
                s=ds.individuals[ind_idx].item(),
                fontsize=8,
                color=color_cycle[ind_idx % len(color_cycle)],
            )
        # plot confidence markers
        # slc_markers = ds.confidence[:, ind_idx] <= 0.1  # there shouldnt be!
        # if any(slc_markers):
        #     ax.scatter(
        #         x=ds.position[slc_markers, ind_idx, 0],
        #         y=ds.position[slc_markers, ind_idx, 1],
        #         s=10,
        #         marker="o",
        #         c="red",
        #     )

    ax.set_aspect("equal")
    ax.set_ylim(-150, 2500)
    ax.set_xlabel("x (pixels)")
    ax.set_ylabel("y (pixels)")
    ax.set_title(title)
    ax.invert_yaxis()
    plt.show()

# %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
# plot predicted trajectories and color by confidence
# ds.position ---> time, individuals, space

ds = ds_pred
title = "Prediction - color by confidence of detection"

for vmin in [0.0, 0.8]:
    fig, ax = plt.subplots(1, 1)
    for ind_idx in range(ds.sizes["individuals"]):
        im = ax.scatter(
            x=ds.position[:, ind_idx, 0],  # nframes, nindividuals, x
            y=ds.position[:, ind_idx, 1],
            s=1,
            c=ds.confidence[:, ind_idx],
            cmap="viridis",  # Optional: change colormap if desired
            vmin=vmin,  # np.nanmin(ds.confidence.data), # Specify the minimum value for the colormap
            vmax=1.0,  # np.nanmax(ds.confidence.data), # Specify the maximum value for the colormap
        )
    ax.set_aspect("equal")
    ax.invert_yaxis()
    ax.set_xlabel("x (pixels)")
    ax.set_ylabel("y (pixels)")
    ax.set_title(title)

    # Add a colorbar based on the scatter plot
    cbar = fig.colorbar(im, ax=ax)
    cbar.set_label("Confidence")  # Optional: label for the colorbar

    plt.show()
# %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
# plot first 10 individuals in confidence
fig, ax = plt.subplots(1, 1)

ax.scatter(x=ds_pred.position[:, :10, 0], y=ds_pred.position[:, :10, 1], s=1)
ax.set_aspect("equal")
ax.set_xlabel("x (pixels)")
ax.set_ylabel("y (pixels)")
# %%%%%%%%%%%%%%%%
# groupby
# It generates a break or new group every time the value of the key function
# changes
# input = (
#   np.isnan(ds_gt.position.data[:,0,0]*ds_gt.position.data[:,0,1]
#  ).astype(int))
input = [0, 0, 0, 1, 1, 1, 0, 1, 0, 0, 1, 1, 1]
len_per_chunk = [
    (key, len(list(group_iter)))
    for key, group_iter in itertools.groupby(input)
]
len_per_chunk_with_1 = [
    len(list(group_iter))
    for key, group_iter in itertools.groupby(input)
    if key == 1
]
# %%%%%%%%%%%%%%%%%%%%%%%%%
# # Fix ground truth file and save!
# df = pd.read_csv(groundtruth_csv, sep=",", header=0)

# # find duplicates
# list_unique_filenames = list(set(df.filename))
# filenames_to_rep_ID = {}
# for file in list_unique_filenames:
#     df_one_filename = df.loc[df["filename"] == file]

#     list_track_ids_one_filename = [
#         int(ast.literal_eval(row.region_attributes)["track"])
#         for row in df_one_filename.itertuples()
#     ]

#     if len(set(list_track_ids_one_filename)) != len(
#         list_track_ids_one_filename
#     ):
#         # [
#         #     list_track_ids_one_filename.remove(k)
#         #     for k in set(list_track_ids_one_filename)
#         # ]  # there could be more than one duplicate!!!
#         for k in set(list_track_ids_one_filename):
#             list_track_ids_one_filename.remove(k)  # remove first occurrence

#         filenames_to_rep_ID[file] = list_track_ids_one_filename

# # delete duplicate rows
# for file, list_rep_ID in filenames_to_rep_ID.items():
#     for rep_ID in list_rep_ID:
#         # find repeated rows for selected file and rep_ID
#         matching_rows = df[
#             (df["filename"] == file)
#             & (df["region_attributes"] == f'{{"track":"{rep_ID}"}}')
#         ]

#         # Identify the index of the first matching row
#         if not matching_rows.empty:
#             indices_to_drop = matching_rows.index[1:]

#             # Drop all but the first matching row
#             df = df.drop(indices_to_drop)

# # save to csv
# groundtruth_csv_corrected = Path(groundtruth_csv).parent / Path(
#     Path(groundtruth_csv).stem + "_corrected.csv"
# )
# df.to_csv(groundtruth_csv_corrected, index=False)
