"""Histograms for the crab dataset."""

# %%
from pathlib import Path

import matplotlib.colors as mcolors
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import plotly.io as pio
import xarray as xr

# Hide attributes globally
xr.set_options(
    display_expand_attrs=False,
    display_style="html",  # "text" for readibility in dark mode?
)


pio.renderers.default = "browser"

# %%
# %matplotlib widget
# %matplotlib osx

# %%%%%%%%%%%%%%%%
# Input data


data_dir = Path("/Users/sofia/swc/CrabTracks")
crabs_zarr_dataset = data_dir / "CrabTracks-slurm2412462-slurm2423692.zarr"

data_vars_order = [
    "position",
    "shape",
    "confidence",
    "escape_state",
]

image_w = 4096
image_h = 2160

fps = 59.94

# NOTE: Filter by length, using a fraction
# of clip length?
min_frames_per_trajectory = 60 * 3  # video is 59.94 fps

# %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
# Read dataset as an xarray datatree

dt = xr.open_datatree(
    crabs_zarr_dataset,
    engine="zarr",
    chunks={},
)

# %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
# Histogram clips length
min_clip_length_in_min = 0
max_clip_length_in_min = 45
n_bins = 100

bins = np.linspace(min_clip_length_in_min, max_clip_length_in_min, n_bins)
final_hist_counts_clips = np.zeros(len(bins) - 1)
final_hist_counts_escapes = np.zeros(len(bins) - 1)
final_hist_counts_non_escapes = np.zeros(len(bins) - 1)
for node in dt.leaves:
    # Get clip lengths in frames
    clip_lengths_in_frames = (
        node.ds.clip_last_frame_0idx - node.ds.clip_first_frame_0idx
    ).values + 1

    escape_lengths_in_frames = (
        node.ds.clip_last_frame_0idx - node.ds.clip_escape_first_frame_0idx
    ).values + 1  # both start and end included

    non_escape_lengths_in_frames = (
        clip_lengths_in_frames - escape_lengths_in_frames
    )

    # Compute histograms in minutes
    hist_counts_clips, _ = np.histogram(
        clip_lengths_in_frames / fps / 60, bins=bins, density=False
    )
    hist_counts_escapes, _ = np.histogram(
        escape_lengths_in_frames / fps / 60, bins=bins, density=False
    )
    hist_counts_non_escapes, _ = np.histogram(
        non_escape_lengths_in_frames / fps / 60, bins=bins, density=False
    )

    # Accumulate
    final_hist_counts_clips += hist_counts_clips
    final_hist_counts_escapes += hist_counts_escapes
    final_hist_counts_non_escapes += hist_counts_non_escapes


# plot
fig, ax = plt.subplots(2, 2, figsize=(12, 8))

# clip length
ax[0, 0].bar(
    (bins[:-1] + bins[1:]) / 2,
    final_hist_counts_clips,
    width=bins[1:] - bins[:-1],
    edgecolor="black",
    color="blue",
    label="clip",
)

# escape
ax[1, 0].bar(
    (bins[:-1] + bins[1:]) / 2,
    final_hist_counts_escapes,
    width=bins[1:] - bins[:-1],
    edgecolor="black",
    color="orange",
    label="escape",
)

# non-escape
ax[0, 1].bar(
    (bins[:-1] + bins[1:]) / 2,
    final_hist_counts_non_escapes,
    width=bins[1:] - bins[:-1],
    edgecolor="black",
    color="green",
    label="non-escape",
)

for a in ax.flatten():
    a.set_xlabel("length (min)")
    a.set_ylabel("count")
    a.grid(alpha=0.25)
    a.legend()
    a.set_ylim(0, 50)
    # ax.set_xlim(0, 7000)

ax[1, 0].set_ylim(0, 220)

# remove empty subplot
fig.delaxes(ax[1, 1])

# %%%%%%%%%%%%%%%%%%%
# Number of tracklets (individuals) per clip





# Why is
# (~ds.confidence.isnull()).any(dim='time').sum(dim='individuals').values.max != len(ds.individuals?





# %%%%%%%%%%%%%%%%%%%%%%%%
# Histogram number of samples per individual

# TODO review bins
bins = np.linspace(0, 50_000, 500)
final_hist_counts_clips = np.zeros(len(bins) - 1)
for node in dt.leaves:
    samples = (
        (~node.ds.position.isnull().compute()).any(dim="space").sum(dim="time")
    )
    hist_counts, bin_edges = np.histogram(samples, bins=bins, density=False)

    final_hist_counts_clips += hist_counts

# %%
# plot as a bar plot?
bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2
plt.figure(figsize=(10, 6))
plt.bar(
    bin_centers,
    final_hist_counts_clips,
    width=bin_edges[1] - bin_edges[0],
    edgecolor="black",
)
plt.xlabel("Number of samples per individual")
plt.ylabel("Count")
plt.title("Histogram of samples per individual across all videos")
plt.grid(axis="y", alpha=0.75)
plt.ylim(0, 17500)
plt.xlim(0, 7000)

# %%
plt.figure()
plt.plot(bin_centers, final_hist_counts_clips)
plt.xlabel("n samples in tracklet")
plt.ylabel("Count")
# %%
