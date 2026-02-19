"""Demo notebook for working with crab dataset.



Useful references:
- https://docs.xarray.dev/en/latest/api/datatree.html

"""

# %%
# %%
import io
from pathlib import Path

import datashader as ds
import datashader.transfer_functions as tf
import matplotlib.colors as mcolors
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import plotly.io as pio
import xarray as xr
from PIL import Image

# Hide attributes globally
xr.set_options(
    display_expand_attrs=False,
    display_style="html",  # "text" for readibility in dark mode?
)


pio.renderers.default = "browser"

# %%
# %matplotlib widget
%matplotlib osx

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

min_frames_per_trajectory = 60

# %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
# Read dataset as an xarray datatree

dt = xr.open_datatree(
    crabs_zarr_dataset,
    engine="zarr",
    chunks={},
)

print(dt)


# %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
# Visualise clips in video as a bar plot

# .to_dataset(): makes a copy
# .ds(): returns a view, changes propagate to tree
ds_video = dt["06.09.2023-01-Right"].ds

# prepare data for plot
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
# plots each clip of a video in the browser

ds_video = dt["06.09.2023-01-Right"].ds


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
for clip_id in range(ds_video.clip_id.shape[0]):
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

    # # Filter out short trajectories
    # n_samples_per_individual = (
    #     ds_clip.position.notnull().all(dim="space").sum(dim="time").compute()
    # )
    # ds_clip = ds_clip.sel(
    #     individuals=n_samples_per_individual >= min_frames_per_trajectory * 100
    #     # the mask needs to computed concretely to determine which
    #     # individuals to keep. If we don't compute n_samples_per_individual
    #     # explicitly, it will trigger loading the full ds_clip.position into RAM
    # )

    # Get data
    position = ds_clip.position.values  # (time, space, individuals)
    escape = ds_clip.escape_state.values  # (time,)
    individuals = ds_clip.individuals.values.astype(str)

    # Convert to dataframe (better for plotly)
    df_out = _to_df(position, escape == 0.0, individuals)
    df_in = _to_df(position, escape == 1.0, individuals)
    df_start_per_individual = df_out.groupby("ind", sort=False).first()
    del position

    all_individuals = np.union1d(
        df_out["ind"].unique(),
        df_in["ind"].unique(),
    )  # return individuals in either

    # Build figure
    fig = go.Figure()
    for ind_label in all_individuals:
        color = color_map[ind_label]

        # outbound for this individual
        mask_out_ind = df_out["ind"] == ind_label
        if mask_out_ind.any():
            fig.add_trace(
                go.Scattergl(
                    x=df_out.loc[mask_out_ind, "x"],
                    y=df_out.loc[mask_out_ind, "y"],
                    mode="markers",
                    marker=dict(size=4, color=color),
                    name=ind_label,
                    legendgroup=ind_label,
                    showlegend=True,
                )
            )

        # inbound for this individual
        mask_in_ind = df_in["ind"] == ind_label
        if mask_in_ind.any():
            fig.add_trace(
                go.Scattergl(
                    x=df_in.loc[mask_in_ind, "x"],
                    y=df_in.loc[mask_in_ind, "y"],
                    mode="markers",
                    marker=dict(
                        size=4, color=color, line=dict(color="red", width=0.75)
                    ),
                    name=ind_label,
                    legendgroup=ind_label,
                    showlegend=False,
                )
            )

        # first-position star
        if ind_label in df_start_per_individual.index:
            row = df_start_per_individual.loc[ind_label]
            fig.add_trace(
                go.Scattergl(
                    x=[row["x"]],
                    y=[row["y"]],
                    mode="markers",
                    marker=dict(
                        size=12,
                        symbol="star",
                        color=color,
                        line=dict(color="black", width=0.5),
                    ),
                    name=ind_label,
                    legendgroup=ind_label,
                    showlegend=False,
                )
            )

    # # outbound
    # # One trace per individual
    # for ind_label in np.unique(df_out["ind"]):
    #     mask = df_out["ind"] == ind_label
    #     fig.add_trace(
    #         go.Scattergl(
    #             x=df_out.loc[mask, "x"],
    #             y=df_out.loc[mask, "y"],
    #             mode="markers",
    #             marker=dict(size=4, color=color_map[ind_label]),
    #             name=ind_label,
    #             legendgroup=ind_label,
    #             showlegend=True,
    #         )
    #     )

    # # add marker for first position
    # for ind_label in df_start_per_individual.index:
    #     fig.add_trace(
    #         go.Scattergl(
    #             x=[df_start_per_individual.loc[ind_label, "x"]],
    #             y=[df_start_per_individual.loc[ind_label, "y"]],
    #             mode="markers",
    #             marker=dict(
    #                 size=12,
    #                 symbol="star",
    #                 color=color_map[ind_label],
    #                 line=dict(color="black", width=0.5),
    #             ),
    #             name=ind_label,
    #             legendgroup=ind_label,
    #             showlegend=False,  # avoid duplicate legend entries
    #         )
    #     )

    # # inbound
    # # One trace per individual
    # for ind_label in np.unique(df_in["ind"]):
    #     mask = df_in["ind"] == ind_label
    #     fig.add_trace(
    #         go.Scattergl(
    #             x=df_in.loc[mask, "x"],
    #             y=df_in.loc[mask, "y"],
    #             mode="markers",
    #             marker=dict(
    #                 size=4,
    #                 color=color_map[ind_label],
    #                 line=dict(color="red", width=0.75),  # red edge = inbound
    #             ),
    #             name=ind_label,
    #             legendgroup=ind_label,
    #             showlegend=False,  # avoid duplicate legend entries
    #         )
    #     )

    fig.update_layout(
        title=(
            f"{ds_video.video_id} - "
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
# TODO: add colorbar and count!

ds_video = dt["07.09.2023-04-Right"].ds

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
# Plot occupancy heatmap for all videos of a single day
# TODO: add colorbar and count!

leaves_pattern = "07.09.2023*"
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
    # we use node.ds to get a view (no copy) and extract values
    position = node.ds.position.values
    x = position[..., 0, :].reshape(-1)
    y = position[..., 1, :].reshape(-1)

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
# Plot multi-figure with occupancy heatmap for all days
# (matplotlib)

day_month = ["09.08", "10.08", "04.09", "05.09", "06.09", "07.09"]

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
        3, 2, figsize=(12, 14), gridspec_kw={"hspace": 0.5}
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
        x = position[..., 0, :].reshape(-1)  # returns a flat view
        y = position[..., 1, :].reshape(-1)

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
# Plot all trajectories from one video
# (we use datashader to rasterise because ~7M points)

dynspread_threshold = 1.0
n_frames_in_30min = 30 * 60 * 59.94  # video is 59.94 fps
for dt_video in dt.match("07.09.2023*").leaves[0:1]:  
    ds_video = dt_video.ds

    # # select clips starting before 30min only
    # mask_first_30min = (ds_video.clip_first_frame_0idx <= n_frames_in_30min).compute()  
    # ds_video = ds_video.where(mask_first_30min, drop=True)
    # del mask_first_30min

    # prepare data
    x = ds_video.position.sel(space="x").values.reshape(-1)
    y = ds_video.position.sel(space="y").values.reshape(-1)
    mask = ~np.isnan(x) & ~np.isnan(y)
    x_clean, y_clean = x[mask], y[mask]
    del x, y, mask

    # initialise canvas
    canvas = ds.Canvas(
        plot_width=image_w,
        plot_height=image_h,
        x_range=(0, image_w),
        y_range=(0, image_h),
    )

    # add points to the canvas and rasterise
    agg = canvas.points(pd.DataFrame({"x": x_clean, "y": y_clean}), "x", "y")
    img = tf.shade(agg, cmap=["#1f77b4"])
    img = tf.dynspread(img, threshold=dynspread_threshold)

    # Convert img to PIL and then to bytes for plotly
    img_transposed = img.to_pil().transpose(Image.FLIP_TOP_BOTTOM)
    img_buffer = io.BytesIO()
    img_transposed.save(img_buffer, format="PNG")
    img_buffer.seek(0)

    # Build figure
    fig = go.Figure()
    fig.add_layout_image(
        source=Image.open(img_buffer),
        xref="x",
        yref="y",
        x=0,
        y=0,  # top-left corner in data coords (y is inverted)
        sizex=image_w,
        sizey=image_h,
        sizing="stretch",
        layer="below",
    )

    fig.update_layout(
        title=(
            f"{ds_video.video_id} "
            f"({ds_video.clip_last_frame_0idx.max().compute() / ds_video.fps / 60:.1f} min)"
        ),
        xaxis_title="x (pixels)",
        yaxis_title="y (pixels)",
        yaxis_scaleanchor="x",
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
            range=[image_h, 0],  # inverted
        ),
    )

    fig.show(renderer="browser")


# %%%%%%%%%%%%%
# Other:
# - compute path length per individual?
# - filter by pathlength?
# - plot all distances to starting point (e.g. if starting point in burrow)
#   for all "loop00"
