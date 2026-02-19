"""Demo notebook for working with crab dataset.



Useful references:
- https://docs.xarray.dev/en/latest/api/datatree.html

"""

# %%
from pathlib import Path

import matplotlib.colors as mcolors
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import plotly.io as pio
import psutil
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
crabs_zarr_dataset = (
    data_dir / "CrabTracks-slurm2370554-slurm2382788.zarr"
)  # "CrabTracks-slurm2412462-slurm2423692.zarr"

data_vars_order = [
    "position",
    "shape",
    "confidence",
    "escape_state",
]

image_w = 4096
image_h = 2160

min_frames_per_trajectory = 60

# %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
# Read dataset as an xarray datatree

dt = xr.open_datatree(
    crabs_zarr_dataset,
    engine="zarr",
    chunks={},
)

print(dt)

# %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
# Inspect datatree
print(f"Depth: {dt.depth}")

# Number of groups (i.e. videos)
print(f"Number of groups: {len(dt.groups)}")

# Print clips per video
for ds_video in dt.leaves:
    print(f"{ds_video.path}: {len(ds_video.clip_id)} clips")


# %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
# Inspect a single video
ds_video = dt["04.09.2023-01-Right"].to_dataset()
# reorder data vars
ds_video = ds_video[data_vars_order]

# Dimensions, coordinates and data vars
# Untracked coordinates
# shows dask arrays unloaded
print(ds_video)

# %%
# Load coordinates only
# (any coord will do, we choose clip_escape_first_frame_0idx)
# Check clip_id metadta

# .load(): applies in-place,
# .compute(): returns a new object
ds_video.coords["clip_escape_first_frame_0idx"].load()
print(ds_video)


# %%
# Check video dataset attributes
print("Dataset attributes:")
print(*ds_video.attrs.items(), sep="\n")


# %%
# Load all data
# Compare memory before and after loading dask arrays into memory.
# Check memory before
# process = psutil.Process(os.getpid())
# mem_before = process.memory_info().rss / 1_000_000_000
# print(f"Memory before: {mem_before:.2f} GB")

# # Load the data
# ds = dt["04.09.2023-01-Right"]
# ds_loaded = ds.load()

# # Check memory after
# mem_after = process.memory_info().rss / 1_000_000_000
# print(f"Memory after: {mem_after:.2f} GB")


# %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
# Inspect clips in video

# Number of clips in this video
print("Number of clips:", ds_video.coords["clip_id"].size)

# Types of escape
print("Escape types:", np.unique(ds_video.clip_escape_type.values))

# Number of triggered and spontaneous clips
ds_video_triggered = ds_video.where(
    ds_video.clip_escape_type == "triggered", drop=True
)
ds_video_spontaneous = ds_video.where(
    ds_video.clip_escape_type == "spontaneous", drop=True
)

print("Number of triggered clips:", ds_video_triggered.coords["clip_id"].size)
print(
    "Number of spontaneous clips:", ds_video_spontaneous.coords["clip_id"].size
)

# %%
# Get escape type for loop at index 1
print(ds_video.isel(clip_id=1).clip_escape_type.values.item())

# Get escape type for 'Loop09'
print(ds_video.sel(clip_id="Loop09").clip_escape_type.values.item())

# %%
# Visualise clips in video as a bar plot
# prepare for plot
list_colors = [plt.get_cmap("tab20c").colors[i] for i in [2, 3]]  # 2 colors
escape_colors = {
    "triggered": "k",
    "spontaneous": "r",
}
bar_height_plot = 1
bar_widths = (
    ds_video.clip_last_frame_0idx.values
    - ds_video.clip_first_frame_0idx.values
) + 1
bar_edges = ds_video.clip_first_frame_0idx.values

# plot horizontal bar plot
fig, ax = plt.subplots(1, 1, figsize=(8, 5))
rects = ax.barh(
    y=np.zeros_like(bar_widths),
    width=bar_widths,
    left=bar_edges,
    height=bar_height_plot,
    color=list_colors,
)
# add vertical lines for escape frames
ax.vlines(
    ds_video.clip_escape_first_frame_0idx.values,
    ymin=-bar_height_plot / 2,
    ymax=bar_height_plot / 2,
    color=[
        escape_colors[escape_type]
        for escape_type in ds_video.clip_escape_type.values
    ],
    linestyle="--",
)
ax.set_xlim(0, ds_video.clip_last_frame_0idx.values.max())
ax.set_ylim(
    -bar_height_plot / 2,
    bar_height_plot / 2,
)
ax.set_xlabel("frames")
ax.yaxis.set_visible(False)

ax.set_aspect(25_000)
ax.set_title(ds_video.video_id)


# %%%%%%%%%%%%%%%%%%%%%%%%%%
# Visualise trajectories for one clip

ds_video = dt["06.09.2023-01-Right"].to_dataset()


# Helper function
def _to_df(pos, time_mask, individuals):
    position_time_mask = pos[time_mask]  # pos is freed from RAM on return
    non_nan = ~np.isnan(position_time_mask[:, 0, :])  # where x is not nan
    individuals_time_mask = np.broadcast_to(
        individuals,
        (position_time_mask.shape[0], len(individuals)),
    )
    return pd.DataFrame(
        {
            "x": position_time_mask[:, 0, :][non_nan],
            "y": position_time_mask[:, 1, :][non_nan],
            "ind": individuals_time_mask[non_nan],
        }
    )


# Plot one clip per window
for clip_id in [3]:  # range(ds_video.clip_id.shape[0]):
    # Select a clip
    ds_clip = ds_video.isel(clip_id=clip_id)

    # Define colormap (before filtering for easier comparison)
    individuals_all = ds_clip.individuals.values.astype(str)  
    tab20 = plt.cm.tab20.colors  # 20 RGBA tuples
    tab20_hex = [mcolors.to_hex(c) for c in tab20]
    color_map = {
        str(ind): tab20_hex[i % len(tab20_hex)]
        for i, ind in enumerate(individuals_all)
    }

    # Filter out short trajectories
    n_samples_per_individual = (
        ds_clip.position.notnull().all(dim="space").sum(dim="time").compute()
    )
    ds_clip = ds_clip.sel(
        individuals=n_samples_per_individual >= min_frames_per_trajectory * 100
        # the mask needs to computed concretely to determine which
        # individuals to keep. If we don't compute n_samples_per_individual
        # explicitly, it will trigger loading the full ds_clip.position into RAM
    )

    # Get data
    position = ds_clip.position.values  # (time, space, individuals)
    escape = ds_clip.escape_state.values  # (time,)
    individuals = ds_clip.individuals.values.astype(str)

    # Convert to dataframe (better for plotly)
    df_out = _to_df(position, escape == 0.0, individuals)
    df_in = _to_df(position, escape == 1.0, individuals)
    del position

    # Build figure
    fig = go.Figure()

    # outbound
    # One trace per individual
    for ind_label in np.unique(df_out["ind"]):
        mask = df_out["ind"] == ind_label
        fig.add_trace(
            go.Scattergl(
                x=df_out.loc[mask, "x"],
                y=df_out.loc[mask, "y"],
                mode="markers",
                marker=dict(size=4, color=color_map[ind_label]),
                name=ind_label,
                legendgroup=ind_label,
                showlegend=True,
            )
        )

    # inbound
    # One trace per individual
    for ind_label in np.unique(df_in["ind"]):
        mask = df_in["ind"] == ind_label
        fig.add_trace(
            go.Scattergl(
                x=df_in.loc[mask, "x"],
                y=df_in.loc[mask, "y"],
                mode="markers",
                marker=dict(
                    size=4,
                    color=color_map[ind_label],
                    line=dict(color="red", width=0.75),  # red edge = inbound
                ),
                name=ind_label,
                legendgroup=ind_label,
                showlegend=False,  # avoid duplicate legend entries
            )
        )

    # add marker for first position
    first_pos = df_out.groupby("ind", sort=False).first()
    fig.add_trace(
        go.Scattergl(
            x=first_pos["x"],
            y=first_pos["y"],
            mode="markers",
            marker=dict(
                size=12,
                symbol="star",
                color=[color_map[ind] for ind in first_pos.index],
                line=dict(color="black", width=0.5),
            ),
            name="first position",
        )
    )

    fig.update_layout(
        title=(
            f"{ds_video.video_id}"
            f"{ds_clip.clip_id.values.item()} - "
            f"{ds_clip.clip_escape_type.values.item()}"
        ),
        xaxis_title="x (pixels)",
        yaxis_title="y (pixels)",
        yaxis_scaleanchor="x",  # equiv to set_aspect("equal")
        legend_title="Individual",
        plot_bgcolor="white",
        paper_bgcolor="white",
        xaxis=dict(
            showgrid=True,
            gridcolor="lightgrey",
            gridwidth=0.5,
            zeroline=False,
            linecolor="black",
            mirror=True,
            ticks="outside",
            range=[0, image_w],
        ),
        yaxis=dict(
            showgrid=True,
            gridcolor="lightgrey",
            gridwidth=0.5,
            zeroline=False,
            linecolor="black",
            mirror=True,
            ticks="outside",
            range=[image_h, 0],  # invert y-axis
        ),
    )

    fig.show()


# %%%%%%%%%
# Plot occupancy heatmap for all clips in same video
ds_video = dt["07.09.2023-04-Right"].to_dataset()

# prepare data
# flatten to just time and space coords
position_x = ds_video.position.sel(space="x").values
position_y = ds_video.position.sel(space="y").values
mask = ~np.isnan(position_x) & ~np.isnan(position_y)
position_flat_x = position_x[mask]  # returns a flat array
position_flat_y = position_y[mask]

# free original arrays
del position_x, position_y

# plot for all clips in video
fig, ax = plt.subplots()
ax.hist2d(
    position_flat_x,
    position_flat_y,
    bins=[200, 100],
    cmap="viridis",
)
ax.invert_yaxis()
ax.set_xlabel("x (pixels)")
ax.set_ylabel("y (pixels)")
ax.set_aspect("equal")
ax.set_title(ds_video.video_id)

# %%%%%%%%%%%%%%%%%%%%%%%%%%%
# Plot occupancy for a single day
leaves_pattern = "07.09.2023*"  # "07.09.2023*" crashes :(
dt_subset_videos = dt.match(leaves_pattern)

# We build histogram incrementally to avoid RAM peaks
# initialise final histogram data
n_bins_x = 200
n_bins_y = 100
bins = [
    np.linspace(0, image_w, n_bins_x + 1),
    np.linspace(0, image_h, n_bins_y + 1),
]
final_hist = np.zeros((n_bins_x, n_bins_y))

# Add histogram form each video
for node in dt_subset_videos.leaves:
    position = node.ds.position.values  # loads one video
    x = position[..., 0, :].ravel()
    y = position[..., 1, :].ravel()

    del position  # free immediately

    mask = ~np.isnan(x) & ~np.isnan(y)
    video_hist, _, _ = np.histogram2d(x[mask], y[mask], bins=bins)
    final_hist += video_hist

    del x, y, mask

# plot final histogram
fig, ax = plt.subplots(1, 1)
ax.pcolormesh(
    bins[0],
    bins[1],
    final_hist.T,
    cmap="viridis",
)
ax.invert_yaxis()
ax.set_xlabel("x (pixels)")
ax.set_ylabel("y (pixels)")
ax.set_aspect("equal")
ax.set_title(
    f"All videos matching {leaves_pattern} (n={len(dt_subset_videos)})"
)


# %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
# Plot occupancy heatmap for videos in multiple days

day_month = [
    "09.08",
    "10.08",
    "04.09",
    "05.09",
    "06.09",
    "07.09",
]

# image size is 4096 × 2160 pixels
image_w = 4096
image_h = 2160
n_bins_x = 200
n_bins_y = 100
bins = [
    np.linspace(0, image_w, n_bins_x + 1),
    np.linspace(0, image_h, n_bins_y + 1),
]

# multi-plot figure
if len(day_month) > 1:
    fig, axs = plt.subplots(
        3,
        2,
        figsize=(12, 14),
        gridspec_kw={"hspace": 0.5},
    )
else:
    fig, axs = plt.subplots(1, 1)

for d_i, date in enumerate(day_month):
    leaves_pattern = f"{date}.2023*"
    dt_subset_videos = dt.match(leaves_pattern)

    # We build histogram incrementally to avoid RAM peaks
    # initialise final histogram data
    final_hist = np.zeros((n_bins_x, n_bins_y))

    # Add histogram form each video
    for node in dt_subset_videos.leaves:
        position = node.ds.position.values  # loads one video
        x = position[..., 0, :].ravel()  # returns a flat view
        y = position[..., 1, :].ravel()

        del position  # free immediately

        mask = ~np.isnan(x) & ~np.isnan(y)
        video_hist, _, _ = np.histogram2d(x[mask], y[mask], bins=bins)
        final_hist += video_hist

        del x, y, mask

    # plot
    ax = axs.flatten()[d_i] if len(day_month) > 1 else axs
    ax.pcolormesh(
        bins[0],
        bins[1],
        final_hist.T,
        cmap="viridis",
    )
    ax.invert_yaxis()
    ax.set_xlabel("x (pixels)")
    ax.set_ylabel("y (pixels)")
    ax.set_aspect("equal")
    ax.set_title(
        f"All videos matching {leaves_pattern} (n={len(dt_subset_videos)})"
    )


# %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
# plot all trajectories in one video
fig, ax = plt.subplots()
ax.scatter(
    position_flat.sel(space="x").values,
    position_flat.sel(space="y").values,
)
ax.invert_yaxis()
ax.set_xlabel("x (pixels)")
ax.set_ylabel("y (pixels)")
ax.set_aspect("equal")


# %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
# Compute path length per individual


# %%
# Select one clip
ds_clip = ds_video2.isel(clip_id=0)
# or equivalently: ds_clip = ds_video2.sel(clip_id="Loop00")


# to show stats of confidence values per clip
ds_clip.confidence.mean().compute()
ds_clip.confidence.std().compute()
ds_clip.confidence.min().compute()
ds_clip.confidence.max().compute()
ds_clip.confidence.median(dim=("individuals")).compute()

# for all video
ds_video2.confidence.median(dim=("time", "individuals")).compute()


# %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
# Plot all individuals one clip
# video_name = "04.09.2023-01-Right"


# %%%%%%%%%%%%%
# Datatree Operations

# Most xarray computation methods also exist
# as methods on datatree objects
dt.mean(dim=["time", "individuals"])
dt.std(dim="time").compute()  # .compute to get actual values

# Datatree with all first clips of all videos
# The arguments passed to the method are used for every node,
# so the values of the arguments you pass might be valid
# for one node and invalid for another
dt.isel(clip_id=0)

# Datatree with the first 100 frames of all clips
dt.sel(time=slice(0, 100))

# Load all first videos per day into one datatree
leaves_pattern = "*01-Right*"
dt_subset_videos = dt.match(leaves_pattern)


# Scale
# Note: this would also scale confidence
scalebar_in_pixels = 100
scalebar_in_mm = 50
dt * (scalebar_in_pixels / scalebar_in_mm)


# Map over datasets
def scale(ds, factor):
    if "position" in ds:
        return ds.assign(position=ds.position * factor)
    return ds


dt.map_over_datasets(scale, 8)


# %%
# To promote a non-dimension coordinate to dimension coordinate

# Make clip_start_frame_0idx the dimension coordinate instead of clip
# ds_reindexed = ds_video.swap_dims({'clip_id': 'clip_start_frame_0idx'})

# Now this works:
# ds_reindexed.sel(clip_start_frame_0idx=79174)

# %%%%%%%%%%%%%
# Other:
# - compute path length per individual?
# - filter by pathlength?
# - plot all distances to starting point (e.g. if starting point in burrow)
#   for all "loop00"
# - what do I do with a datatree...?
