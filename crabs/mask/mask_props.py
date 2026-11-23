"""Measure the masks of a store and write mask and ellipse parameters.

The measuring code is a copy of ``compute_mask_props_dataset`` in
``scripts/generate_masks_from_bboxes.py`` (PR #298). The script is
standalone and does not depend on ``crabs``, and ``crabs`` cannot import
the script, so the code is copied. A unit test checks the two agree.
"""

import logging
import time
from pathlib import Path

import numpy as np
import xarray as xr
import zarr
from skimage.measure import regionprops_table

from crabs.mask.utils.float_encoding import float_var_encoding

# Map regionprops_table property to data variable
REGIONPROPS_TO_VARS = {
    "area": "mask_area",
    "solidity": "mask_area_to_convex_hull_area",
    "extent": "mask_area_to_mask_bbox_area",
    "axis_major_length": "ellipse_major_axis_length",
    "axis_minor_length": "ellipse_minor_axis_length",
    "eccentricity": "ellipse_eccentricity",
}

# Variables with dimensions clip_id, time, individual
SCALAR_VARS = [*REGIONPROPS_TO_VARS.values(), "ellipse_major_axis_theta"]

# All variables written by `compute_mask_props_of_video`
MASK_PROPS_VARS = [*SCALAR_VARS, "mask_centroid"]


def measure_label_frame(label_frame: np.ndarray) -> dict[str, np.ndarray]:
    """Measure every mask in one label image.

    Returns one array per property, with one value per label found in
    the image along the first axis (``"label"`` holds the labels
    themselves). ``"mask_centroid"`` has shape (n_labels, 2), for (x, y).
    Label 0, the background, is skipped by scikit-image.
    """
    table = regionprops_table(
        label_frame,
        properties=["label", "centroid", "orientation", *REGIONPROPS_TO_VARS],
    )
    out = {"label": table["label"]}
    out |= {var: table[prop] for prop, var in REGIONPROPS_TO_VARS.items()}

    # Convert geometric properties to the x-axis
    # (scikit-image measures from the row axis)
    out["ellipse_major_axis_theta"] = np.mod(
        np.pi / 2 - table["orientation"], np.pi
    )
    # (x, y) = (column, row)
    out["mask_centroid"] = np.stack(
        [table["centroid-1"], table["centroid-0"]], axis=-1
    )
    return out


def get_frames_idcs_with_boxes(ds_clip: xr.Dataset) -> np.ndarray:
    """Return the `time` indices where any individual has non-nan position."""
    any_box = ds_clip.position.notnull().any(dim=["space", "individual"])
    return np.flatnonzero(any_box.values)


def compute_clip_mask_props_dataset(
    labels: xr.DataArray,
    frames_with_boxes: np.ndarray,
    label_of: np.ndarray,
) -> xr.Dataset:
    """Measure the masks of one clip.

    `labels` is read one time chunk at a time, and only the frames in
    `frames_with_boxes` are read, so chunks with none are skipped.

    Pass `labels` lazily, e.g. ``ds.labels.isel(clip_id=c)`` of a dataset
    opened with `xr.open_zarr`: loading a whole clip of 4K label images
    would take several GB.

    The values measured for label ``label_of[k]`` are placed at position
    ``k`` along `individual`.

    The ellipse is the one with the same second moments as the mask, as
    defined by scikit-image. The returned dataset has these float32 data
    variables (NaN where there is no mask):

    - ``mask_centroid`` (time, space, individual): mean (x, y) of the mask
      pixels, which is also the ellipse centre.
    - ``mask_area``: number of mask pixels.
    - ``mask_area_to_convex_hull_area``: mask area / area of its convex
      hull (scikit-image's ``solidity``).
    - ``mask_area_to_mask_bbox_area``: mask area / area of the mask's own
      axis-aligned bounding box (scikit-image's ``extent``). Note this is
      the bounding box of the mask, not the tracked (prompt) bounding
      box.
    - ``ellipse_major_axis_length``, ``ellipse_minor_axis_length``: full
      axis lengths in pixels, not semi-axes.
    - ``ellipse_major_axis_theta``: angle of the major axis from the image
      x-axis (columns) towards the y-axis (rows), in radians, in [0, pi).
      The image y-axis points down, so a positive angle looks clockwise
      on screen. It is arbitrary for near-circular masks.
    - ``ellipse_eccentricity``: 0 for a circle, towards 1 for an elongated
      ellipse.

    The scalar variables have dims (time, individual).

    Parameters
    ----------
    labels : xr.DataArray
        The (time, img_h, img_w) label images of the clip.
    frames_with_boxes : np.ndarray
        Positions along the `time` of `labels` to measure, e.g. the
        frames with boxes. The other frames are left NaN.
    label_of : np.ndarray
        Label of each individual, in `individual` order.

    Returns
    -------
    xr.Dataset
        The 8 mask and ellipse parameters for the clip.

    """
    n_frames, n_individuals = labels.sizes["time"], len(label_of)

    # Map each label to its position along `individual` (-1 if none)
    individual_of_label = np.full(int(label_of.max()) + 1, -1)
    individual_of_label[label_of] = np.arange(n_individuals)

    # Initialise one (time, individual) array per scalar variable, and a
    # (time, individual, space) array for the centroid; NaN where no mask
    shape = (n_frames, n_individuals)
    arrays = {var: np.full(shape, np.nan, np.float32) for var in SCALAR_VARS}
    arrays["mask_centroid"] = np.full((*shape, 2), np.nan, np.float32)

    # Loop thru the time chunks of `labels` (one chunk if not lazy)
    frames_with_boxes = np.asarray(frames_with_boxes)
    time_chunks = labels.chunksizes.get("time", (n_frames,))
    chunk_ends = np.cumsum(time_chunks)
    chunk_starts = chunk_ends - time_chunks
    for start, end in zip(chunk_starts, chunk_ends, strict=True):
        # Read only the frames to measure, so chunks without any are
        # skipped (reading one frame decodes its whole chunk anyway)
        frames = frames_with_boxes[
            (frames_with_boxes >= start) & (frames_with_boxes < end)
        ]
        if not len(frames):
            continue

        # Label images
        label_frames = labels.isel(time=frames).values

        for frame_idx, label_frame in zip(frames, label_frames, strict=True):
            props = measure_label_frame(label_frame)

            # Fill in arrays at each label's individual
            # (values are rounded to float32 here)
            individual_idcs = individual_of_label[props["label"]]
            for var, arr in arrays.items():
                arr[frame_idx, individual_idcs] = props[var]

    centroid = arrays.pop("mask_centroid")
    return xr.Dataset(
        {var: (("time", "individual"), arr) for var, arr in arrays.items()}
        | {
            "mask_centroid": (
                ("time", "space", "individual"),
                centroid.transpose(0, 2, 1),
            )
        }
    )


def compute_mask_props_of_video(
    video_id: str, store_path: str | Path, overwrite: bool = False
) -> None:
    """Measure the masks of one masks group and write the 8 variables.

    The video's group must hold the prompt boxes (see
    `copy_video_bboxes_to_masks`). Only frames with a non-nan `position`
    are measured.

    The variables are first created with no data, and then each clip is
    measured and written with one write. This is to bound memory: only one
    clip of values and one time chunk of `labels` are held at a time.

    Every read and write skips the root's consolidated metadata, which may
    be stale while the store is being written.

    Parameters
    ----------
    video_id : str
        Name of the video group.
    store_path : str | Path
        Path to the masks zarr store.
    overwrite : bool
        Whether to replace the 8 variables if they already exist.
        Default: False.

    Raises
    ------
    ValueError
        Before anything is written, if `label_of` has a repeated value or
        a 0 (the background label), if the group has no `position`, or if
        any of the 8 variables exists and `overwrite` is False.

    """
    ds = xr.open_zarr(store_path, group=video_id, consolidated=False)

    # Check inputs
    label_of = ds.label_of.values.astype(np.int64)
    if len(np.unique(label_of)) != len(label_of) or (label_of == 0).any():
        raise ValueError(
            f"{video_id}: `label_of` must hold unique, non-zero labels."
        )
    if "position" not in ds:
        raise ValueError(
            f"{video_id}: the group has no `position`. Copy the boxes with "
            "copy_video_bboxes_to_masks first."
        )

    # If overwrite is False and mask property variables exist: throw error
    existing = [var for var in MASK_PROPS_VARS if var in ds]
    if existing and not overwrite:
        raise ValueError(
            f"{video_id}: the group already holds {existing}. Pass "
            "overwrite=True to replace them."
        )

    # Delete the mask property vars we wish to overwrite in the group
    # (xarray refuses an encoding argument if the variables already exist)
    group = zarr.open_group(store_path, path=video_id, use_consolidated=False)
    for var in existing:
        del group[var]

    # Create the 8 variables, writing metadata only. They take the dims
    # and chunks of the copied `confidence` (scalars) and `position`
    # (centroid). `.variable` leaves out coords, which already exist.
    def nan_like(data_array: xr.DataArray) -> xr.Variable:
        var = xr.full_like(data_array.variable, np.nan, dtype=np.float32)
        var.attrs = {}
        return var

    ds_template = xr.Dataset(
        {var: nan_like(ds.confidence) for var in SCALAR_VARS}
        | {"mask_centroid": nan_like(ds.position)},
        # copy the group's attributes, as to_zarr(mode="a") replaces them
        attrs=ds.attrs,
    )
    ds_template.to_zarr(
        store_path,
        group=video_id,
        mode="a",
        encoding={
            var: float_var_encoding(ds_template[var].dims)
            for var in MASK_PROPS_VARS
        },
        compute=False,
        consolidated=False,
    )

    # Loop thru clips
    for clip_index, clip_id in enumerate(ds.clip_id.values):
        start_time = time.perf_counter()

        # Measure masks of frames with boxes (`labels` stays lazy)
        ds_clip = ds.isel(clip_id=clip_index)
        ds_props = compute_clip_mask_props_dataset(
            ds_clip.labels,
            get_frames_idcs_with_boxes(ds_clip[["position"]]),
            label_of,
        )

        # Write the clip's values
        ds_props.expand_dims("clip_id").to_zarr(
            store_path,
            group=video_id,
            region={"clip_id": slice(clip_index, clip_index + 1)},
            consolidated=False,
        )
        logging.info(
            f"Measured masks of {video_id}/{clip_id} in "
            f"{time.perf_counter() - start_time:.1f} s"
        )
