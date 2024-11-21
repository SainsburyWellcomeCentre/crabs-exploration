"""Inspect crab escape trajectories using movement"""

# %%
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from movement.io import load_bboxes

# %%%%%%%%%%%%%%%%%%%%%%%%
# Requirements
# 1. Install the crabs package with "notebooks" dependencies. 
#    From the root folder of the repo and in an active conda environment run
#       ```
#       pip install .[notebooks]
#       ```
# 2. Mount the zoo directory in ceph following the guide at 
#    https://howto.neuroinformatics.dev/programming/Mount-ceph-ubuntu-temp.html

# %%%%%%%%%%%%%%%%%%%%%%%%
# Uncomment the following line to enable interactive plots
# %matplotlib widget

# %%%%%%%%%%%%%%%%%%%%%
# Input data
# Ensure the input data points to the directory containing the csv files in ceph
input_data = Path(
    "/ceph/zoo/users/sminano/escape_clips_tracking_output_slurm_5699097"
    #"/home/sminano/swc/project_crabs/escape_clips_tracking_output_slurm_5699097"
)

# Get a list of all csv files in that directory
list_csv_files = [x for x in input_data.iterdir() if x.is_file()]
list_csv_files.sort()
print(len(list_csv_files))


# %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
# Read each file as a movement dataset and add to a dictionay
map_file_to_dataset = {}
for csv_file in list_csv_files:
    map_file_to_dataset[csv_file] = load_bboxes.from_via_tracks_file(
        csv_file, fps=None, use_frame_numbers_from_file=False
    )

# %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
# Print summary metrics for one dataset
ds = map_file_to_dataset[list_csv_files[0]]
print(ds)

# Print summary
print(f"{ds.source_file}")
print(f"Number of frames: {ds.sizes['time']}")
print(f"Number of individuals: {ds.sizes['individuals']}")

# %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
# Plot trajectories for one dataset

fig, ax = plt.subplots(1, 1)

# Define colors - ideally more than max n individuals
list_colors = (
    plt.get_cmap("Pastel1").colors  # 9 colors
    + plt.get_cmap("Pastel2").colors  # 8 colors
    + plt.get_cmap("Paired").colors  # 12 colors
    + plt.get_cmap("Accent").colors  # 8 colors
    + plt.get_cmap("Dark2").colors  # 8 colors
    + plt.get_cmap("Set1").colors  # 9 colors
    + plt.get_cmap("Set3").colors  # 12 colors
    + plt.get_cmap("tab20b").colors  # 10 colors
    + plt.get_cmap("tab20c").colors  # 20 colors
)  # 96 colors


for ind_idx in range(ds.sizes["individuals"]):
    # plot trajectories
    ax.scatter(
        x=ds.position[:, ind_idx, 0],  # nframes, nindividuals, x
        y=ds.position[:, ind_idx, 1],
        s=1,
        color=list_colors[ind_idx % len(list_colors)],
    )
    # add ID at first frame with non-nan x-coord
    start_frame = ds.time[~ds.position.isnull()[:, ind_idx, 0]][0].item()
    ax.text(
        x=ds.position[start_frame, ind_idx, 0],
        y=ds.position[start_frame, ind_idx, 1],
        s=ds.individuals[ind_idx].item().split("_")[1],
        fontsize=8,
        color=list_colors[ind_idx % len(list_colors)],
    )

ax.set_aspect("equal")
ax.set_ylim(-150, 2500)
ax.set_xlabel("x (pixels)")
ax.set_ylabel("y (pixels)")
ax.set_title(list_csv_files[0].stem)
ax.invert_yaxis()
plt.show()

# %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
# Check lengths of trajectories

non_nan_frames_pred = {}
for ind, _id_str in enumerate(ds.individuals):
    non_nan_frames_pred[ind] = (
        len(ds.time) - ds.position[:, ind, :].isnull().any(axis=1).sum().item()
    )

fig, ax = plt.subplots(1, 1)
out = ax.hist(
    non_nan_frames_pred.values(),
    bins=np.arange(0, len(ds.time) + 50, 50),
    alpha=0.5,
    label="Prediction",
)
ax.set_xlabel("n frames with same ID")
ax.set_ylabel("n tracks")
ax.tick_params(labeltop=False, labelright=True, right=True, which="both")
ax.hlines(
    y=len(ds.individuals), xmin=0, xmax=len(ds.time), color="red"
)  # n of individuals
ax.legend(bbox_to_anchor=(1.0, 1.16))
ax.set_title(list_csv_files[0].stem)
# %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
# Save plots
