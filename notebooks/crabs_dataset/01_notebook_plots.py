"""Demo notebook for working with crab dataset.



Useful references:
- https://docs.xarray.dev/en/latest/api/datatree.html

"""

# %%
# %%
import io
import re
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
# %matplotlib osx

# %%%%%%%%%%%%%%%%
# Input data


data_dir = Path().home() / "swc" / "project_crabs" / "data" / "CrabTracks"
crabs_zarr_dataset = data_dir / "CrabTracks-slurm2478780-2478861-2489356.zarr"

data_vars_order = [
    "position",
    "shape",
    "confidence",
    "escape_state",
]

image_w = 4096
image_h = 2160

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
ds_video = dt["05.09.2023-05-Right"].ds

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

frames_to_min = 1 / (59.94 * 60)

# plot horizontal bar plot
fig, ax = plt.subplots(1, 1, figsize=(8, 5))
rects = ax.barh(
    y=np.zeros_like(bar_widths),
    width=bar_widths * frames_to_min,
    left=bar_edges * frames_to_min,
    height=bar_height_plot,
    color=list_colors,
)
# add vertical lines for escape frames
ax.vlines(
    ds_video.clip_escape_first_frame_0idx.values * frames_to_min,
    ymin=-bar_height_plot / 2,
    ymax=bar_height_plot / 2,
    color=[
        escape_colors[escape_type]
        for escape_type in ds_video.clip_escape_type.values
    ],
    linestyle="--",
)

# Add text labels to each bar
for i, rect in enumerate(rects):
    text_x = rect.get_x() + rect.get_width() / 2
    text_y = rect.get_y() + rect.get_height() / 2
    ax.text(
        text_x,
        text_y,
        f"({rect.get_width():.1f})",
        va="center",
        ha="center",
        fontsize=10,
        color="black",
    )


ax.set_xlim(0, ds_video.clip_last_frame_0idx.values.max() * frames_to_min)
ax.set_ylim(
    -bar_height_plot / 2,
    bar_height_plot / 2,
)
ax.set_xlabel("time (min)")
ax.yaxis.set_visible(False)

ax.set_aspect(25_000 * frames_to_min)
ax.set_title(ds_video.video_id)


# %%
# If we use an accessor:
@xr.register_datatree_accessor("plot_clips_in_video")
class PlotClipsAccessor:
    def __init__(self, dt):
        self._dt = dt

        self.bar_height = 1
        self.bar_colors = [plt.get_cmap("tab20c").colors[i] for i in [2, 3]]
        self.escape_colors = {
            "triggered": "k",
            "spontaneous": "r",
        }

    def __call__(self, clip_id):
        ds_video = self._dt[clip_id].ds

        # prepare data for plot
        bar_widths = (
            ds_video.clip_last_frame_0idx.values
            - ds_video.clip_first_frame_0idx.values
        ) + 1
        bar_edges = ds_video.clip_first_frame_0idx.values

        frames_to_min = 1 / (ds_video.fps * 60)
        video_dur_min = (
            ds_video.clip_last_frame_0idx.values.max() + 1
        ) * frames_to_min

        # plot horizontal bar plot
        fig, ax = plt.subplots(1, 1, figsize=(8, 5))
        rects = ax.barh(
            y=np.zeros_like(bar_widths),
            width=bar_widths * frames_to_min,
            left=bar_edges * frames_to_min,
            height=self.bar_height,
            color=self.bar_colors,
        )

        # add vertical lines for escape frames
        ax.vlines(
            ds_video.clip_escape_first_frame_0idx.values * frames_to_min,
            ymin=-self.bar_height / 2,
            ymax=self.bar_height / 2,
            color=[
                self.escape_colors[escape_type]
                for escape_type in ds_video.clip_escape_type.values
            ],
            linestyle="--",
        )

        # Add text labels to each bar
        for i, rect in enumerate(rects):
            text_x = rect.get_x() + rect.get_width() / 2
            text_y = rect.get_y() + rect.get_height() / 2
            ax.text(
                text_x,
                text_y,
                f"({rect.get_width():.1f})",
                va="center",
                ha="center",
                fontsize=10,
                color="black",
            )

        # legend for escape line types
        legend_handles = [
            plt.Line2D([0], [0], color=color, linestyle="--", label=label)
            for label, color in self.escape_colors.items()
        ]
        ax.legend(
            handles=legend_handles,
            loc="upper left",
            bbox_to_anchor=(1.0, 1.0),
        )

        ax.set_xlim(0, video_dur_min)
        ax.set_ylim(
            -bar_height_plot / 2,
            bar_height_plot / 2,
        )
        ax.set_xlabel("time (min)")
        ax.yaxis.set_visible(False)

        ax.set_aspect(25_000 * frames_to_min)
        ax.set_title(
            f"{ds_video.video_id} ({video_dur_min:.2f} min, "
            f"n_clips ={len(ds_video.clip_escape_type.values)})"
        )

        return fig, ax


# %%
# Then we can do
# dt.plot_clips("04.09.2023-05-Right")

for video_id in list(dt.match("04.09.2023*")):
    dt.plot_clips_in_video(video_id)

# %%%%%%%%%%%%%%%%%%%%%%%%%%
# Visualise trajectories for one clip
# plots each clip of a video in the browser


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


# @dask.delayed
def plot_clip(clip_id, ds_video):
    # Select a clip
    ds_clip = ds_video.isel(clip_id=clip_id)
    clip_duration_frames = (
        ds_clip.clip_last_frame_0idx - ds_clip.clip_first_frame_0idx
    ).values.item() + 1

    # Define colormap
    individuals_all = ds_clip.individuals.values.astype(str)
    tab20 = plt.cm.tab20.colors  # 20 RGBA tuples
    tab20_hex = [mcolors.to_hex(c) for c in tab20]
    color_map = {
        str(ind): tab20_hex[i % len(tab20_hex)]
        for i, ind in enumerate(individuals_all)
    }

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

    fig.update_layout(
        title=(
            f"{ds_video.video_id} - "
            f"{ds_clip.clip_id.values.item()} - "
            f"{ds_clip.clip_escape_type.values.item()} "
            f"({clip_duration_frames * frames_to_min:.1f} min)"
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

    return fig


# %%
# Plot one clip per browser tab
for clip_id in [6]:  # range(ds_video.clip_id.shape[0]):
    fig = plot_clip(clip_id, dt["06.09.2023-01-Right"].ds)
    fig.show()

    # self-contained HTML (available offline)
    title = fig.layout.title.text or f"clip_{clip_id}"
    safe_title = re.sub(r'[\\/*?:"<>|]', "_", title)
    print(safe_title)
    fig.write_html(
        f"/Users/sofia/arc/project_Zoo_crabs/crabs-exploration/{safe_title}.html",
        include_plotlyjs=True,
    )

# %%
# # Save each plot as an svg
# import re
# from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor


# def save_clip_svg(clip_id):
#     try:
#         dt = xr.open_datatree(
#             crabs_zarr_dataset,
#             engine="zarr",
#             chunks={},
#         )
#         ds_video = dt["06.09.2023-01-Right"].ds

#         fig = plot_clip(clip_id, ds_video)
#         title = fig.layout.title.text or f"clip_{clip_id}"
#         safe_title = re.sub(r'[\\/*?:"<>|]', "_", title)
#         print(safe_title)
#         # Self-contained HTML (includes Plotly.js inline, ~3MB)
#         # fig.write_html("figure.html", include_plotlyjs=True)

#         # Or use a CDN link instead (much smaller file)
#         # requires an internet connection to render.
#         fig.write_html(
#             f"/Users/sofia/arc/project_Zoo_crabs/crabs-exploration/{safe_title}.html",
#             include_plotlyjs="cdn",
#         )
#     except Exception as e:
#         print(f"Error saving clip {clip_id}: {e}")


# with ThreadPoolExecutor(max_workers=8) as executor:
#     executor.map(save_clip_svg, range(ds_video.clip_id.shape[0]))

# # %%
# # Plot in parallel?

# # tasks = [plot_clip(i, ds_video, image_w, image_h) for i in range(ds_video.clip_id.shape[0])]
# # dask.compute(*tasks, scheduler="threads")


# %%%%%%%%%
# Plot occupancy heatmap for all clips in same video
ds_video = dt["06.09.2023-01-Right"].ds

# prepare data
# flatten to just time and space coords
position_x = ds_video.position.sel(space="x").values
position_y = ds_video.position.sel(space="y").values
mask = ~np.isnan(position_x) & ~np.isnan(position_y)
position_flat_x = position_x[mask]  # returns a flat array
position_flat_y = position_y[mask]

# free original arrays
del position_x, position_y

n_bins_x = 100
n_bins_y = round(n_bins_x * (image_h / image_w))

# plot for all clips in video
fig, ax = plt.subplots()
h, _, _, img = ax.hist2d(
    position_flat_x,
    position_flat_y,
    bins=[n_bins_x, n_bins_y],
    cmap="viridis",
)
fig.colorbar(img, ax=ax, label="count")
ax.invert_yaxis()
ax.set_xlabel("x (pixels)")
ax.set_ylabel("y (pixels)")
ax.set_aspect("equal")
ax.set_title(ds_video.video_id)

# %%%%%%%%%%%%%%%%%%%%%%%%%%%
# Plot occupancy heatmap for all videos of a single day

leaves_pattern = "06.09.2023*"
dt_subset_videos = dt.match(leaves_pattern)

# We build histogram incrementally to avoid RAM peaks
# initialise final histogram data
n_bins_x = 100
n_bins_y = round(n_bins_x * (image_h / image_w))
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
    video_hist, _, _ = np.histogram2d(
        x[mask], y[mask], bins=bins, density=False
    )
    final_hist += video_hist

    del x, y, mask

# plot final histogram
fig, ax = plt.subplots(1, 1)
mesh = ax.pcolormesh(
    bins[0],
    bins[1],
    final_hist.T,
    cmap="viridis",
)
fig.colorbar(mesh, ax=ax, label="count")
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
n_bins_x = 200
n_bins_y = round(n_bins_x * (image_h / image_w))
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

dynspread_threshold = 0.975
n_frames_in_30min = 30 * 60 * 59.94  # video is 59.94 fps
for dt_video in dt.match("06.09.2023*").leaves:
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

    # Save as PNG with transparent background (image_w x image_h, RGBA)
    safe_video_id = re.sub(r'[\\/*?:"<>|]', "_", str(ds_video.video_id))
    safe_video_id = safe_video_id.split(" ")[0]  # remove after first space
    img_transposed.save(
        f"/Users/sofia/arc/project_Zoo_crabs/crabs-exploration/{safe_video_id}.png",
        format="PNG",
    )

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
