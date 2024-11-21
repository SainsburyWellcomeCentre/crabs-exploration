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
)

output_figures_dir = Path(
    "/ceph/zoo/users/sminano/escape_clips_tracking_output_slurm_5699097/figures"
)

# %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
# List all csv files in the input directory
list_csv_files = [x for x in input_data.iterdir() if x.is_file()]
list_csv_files.sort()
print(len(list_csv_files))


# %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
# Read each file as a movement dataset and add to a dictionay
# This step is slow
map_file_to_dataset = {}
for csv_file in list_csv_files:
    map_file_to_dataset[csv_file] = load_bboxes.from_via_tracks_file(
        csv_file, fps=None, use_frame_numbers_from_file=False
    )

# %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
# Print summary metrics for one dataset
for _csv_file, ds in map_file_to_dataset.items():
    print(Path(ds.source_file).name)
    print(f"Number of frames: {ds.sizes['time']}")
    print(f"Number of individuals: {ds.sizes['individuals']}")
    print(ds)
    print("--------------------")


# %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
# Generate plots for each dataset

# Create a directory if it doesnt exist
if not output_figures_dir.exists():
    output_figures_dir.mkdir(parents=True)

for _csv_file, ds in map_file_to_dataset.items():

    ##%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    # Plot trajectories for one dataset
    fig, ax = plt.subplots(1, 1)

    # select whether to plot ID at first frame
    flag_plot_id = False

    # Define colors - ideally more than max n individuals
    # so that we don't have repetitions
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
        if flag_plot_id:
            start_frame = ds.time[~ds.position.isnull()[:, ind_idx, 0]][
                0
            ].item()
            ax.text(
                x=ds.position[start_frame, ind_idx, 0],
                y=ds.position[start_frame, ind_idx, 1],
                s=ds.individuals[ind_idx].item().split("_")[1],
                fontsize=8,
                color=list_colors[ind_idx % len(list_colors)],
            )

    ax.set_aspect("equal")
    ax.set_xlim(-150, 4200)  # frame size: 4096x2160
    ax.set_ylim(-150, 2250)  # frame size: 4096x2160
    ax.set_xlabel("x (pixels)")
    ax.set_ylabel("y (pixels)")
    ax.set_title(Path(ds.source_file).stem)
    ax.invert_yaxis()
    plt.savefig(
        output_figures_dir / f"{Path(ds.source_file).stem}_tracks.png",
        dpi=300,
        bbox_inches="tight",
    )

    ##%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    # Plot histogram of trajectories' lengths

    # Compute number of frames with non-nan position per ID
    non_nan_frames_per_ID = {}
    for ind, _id_str in enumerate(ds.individuals):
        non_nan_frames_per_ID[ind] = (
            len(ds.time)
            - ds.position[:, ind, :].isnull().any(axis=1).sum().item()
        )

    # Plot histogram
    fig, ax = plt.subplots(1, 1)
    out = ax.hist(
        non_nan_frames_per_ID.values(),
        bins=np.arange(0, len(ds.time) + 50, 50),
        alpha=0.5,
        label="Prediction",
    )
    ax.set_xlabel("n frames with same ID")
    ax.set_ylabel("n trajectories")
    ax.hlines(
        y=len(ds.individuals),
        xmin=0,
        xmax=len(ds.time),
        color="red",
        label="n individuals",
    )
    ax.legend()
    ax.set_title(Path(ds.source_file).stem)

    # Save plot as png
    plt.savefig(
        output_figures_dir / f"{Path(ds.source_file).stem}_histogram.png",
        dpi=300,
        bbox_inches="tight",
    )
