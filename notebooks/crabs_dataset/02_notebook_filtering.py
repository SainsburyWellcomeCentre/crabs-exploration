


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

# NOTE: Filter by length, using a fraction
# of clip length?
min_frames_per_trajectory = 60*3 # video is 59.94 fps

# %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
# Read dataset as an xarray datatree

dt = xr.open_datatree(
    crabs_zarr_dataset,
    engine="zarr",
    chunks={},
)

print(dt)

# %%%%%%%%%%%%%%%%%%%%%%%%%%%
# Plot all trajectories per clip per video

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

    # -----------------------
    # Filter out short trajectories
    n_samples_per_individual = (
        ds_clip.position.notnull().all(dim="space").sum(dim="time").compute()
    )
    ds_clip = ds_clip.sel(
        individuals=n_samples_per_individual >= min_frames_per_trajectory
        # the mask needs to computed concretely to determine which
        # individuals to keep. If we don't compute n_samples_per_individual
        # explicitly, it will trigger loading the full ds_clip.position into RAM
    )
    # ------------------

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