"""Measure the masks of a masks store and write mask and ellipse parameters.

The measuring code is a copy of ``compute_mask_props_dataset`` in
``scripts/generate_masks_from_bboxes.py`` (PR #298). The script is
standalone and does not depend on ``crabs``, and ``crabs`` cannot import
the script, so the code is copied. A unit test checks the two agree.
"""

import logging
import time
from pathlib import Path

import dask.array as da
import numpy as np
import xarray as xr
import zarr
from skimage.measure import regionprops_table

# Map regionprops_table property to data variable
PROPS_TO_VARS = {
    "area": "mask_area",
    "solidity": "mask_area_to_convex_hull_area",
    "extent": "mask_area_to_mask_bbox_area",
    "axis_major_length": "ellipse_major_axis_length",
    "axis_minor_length": "ellipse_minor_axis_length",
    "eccentricity": "ellipse_eccentricity",
}

# Variables with dims (clip_id, time, individual)
SCALAR_VARS = [*PROPS_TO_VARS.values(), "ellipse_major_axis_theta"]

# All variables written by `add_mask_props_to_group`
MASK_PROPS_VARS = [*SCALAR_VARS, "mask_centroid"]


def measure_label_frame(label_frame: np.ndarray) -> dict[str, np.ndarray]:
    """Measure every mask in one label image.

    Returns one array per property, with one value per label found in
    the image (``"label"`` holds the labels themselves). Label 0, the
    background, is skipped by scikit-image.
    """
    table = regionprops_table(
        label_frame,
        properties=["label", "centroid", "orientation", *PROPS_TO_VARS],
    )
    out = {"label": table["label"]}
    out |= {var: table[prop] for prop, var in PROPS_TO_VARS.items()}

    # Convert geometric properties to the x-axis
    # (scikit-image measures from the row axis)
    out["ellipse_major_axis_theta"] = np.mod(
        np.pi / 2 - table["orientation"], np.pi
    )
    out["mask_centroid_x"] = table["centroid-1"]  # x = column
    out["mask_centroid_y"] = table["centroid-0"]  # y = row
    return out


def frames_with_boxes(ds_clip: xr.Dataset) -> np.ndarray:
    """Return the `time` indices where any individual has a finite position.

    `mask-tracked-crabs` runs SAM2 only on frames with boxes, so every
    other frame of the label images is background.
    """
    any_box = ds_clip.position.notnull().any(dim=["space", "individual"])
    return np.flatnonzero(any_box.values)


def compute_clip_mask_props_dataset(
    labels_array: zarr.Array | np.ndarray,
    clip_index: int,
    time_slice: slice,
    frames: np.ndarray,
    individual_of_label: np.ndarray,
) -> xr.Dataset:
    """Measure the masks of one time chunk of one clip.

    The label images ``labels_array[clip_index, t]`` are read one frame at
    a time, only for the frames ``t`` in `frames` that fall in
    `time_slice`. The values measured for a label are placed at the
    position along `individual` given by ``individual_of_label[label]``,
    the inverse of `label_of`.

    The ellipse is the one with the same second moments as the mask, as
    defined by scikit-image. The returned dataset has no coordinates, and
    these float32 data variables, with NaN where there is no mask:

    - ``mask_centroid`` (time, space, individual): mean (x, y) of the mask
      pixels, which is also the ellipse centre.
    - ``mask_area``: number of mask pixels.
    - ``mask_area_to_convex_hull_area``: mask area / area of its convex
      hull (scikit-image's ``solidity``).
    - ``mask_area_to_mask_bbox_area``: mask area / area of the mask's own
      axis-aligned bounding box (scikit-image's ``extent``). Note this is
      the bounding box of the mask, not the tracked (prompt) bounding
      box, so it depends on the mask's rotation.
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
    labels_array : zarr.Array | np.ndarray
        The (clip_id, time, img_h, img_w) label images.
    clip_index : int
        Position of the clip along `clip_id`.
    time_slice : slice
        The time chunk to measure, with explicit start and stop.
    frames : np.ndarray
        Time indices of the clip to measure, e.g. the frames with boxes.
        Those outside `time_slice` are ignored.
    individual_of_label : np.ndarray
        Position along `individual` of each label, -1 for a value that is
        not a crab's label.

    Returns
    -------
    xr.Dataset
        The 8 mask and ellipse parameters for the time chunk.

    Raises
    ------
    ValueError
        If a label image holds a label that is not in `label_of`.

    """
    n_chunk = time_slice.stop - time_slice.start
    n_individuals = int(individual_of_label.max()) + 1

    # Initialise one (time, individual) array per scalar variable, and a
    # (time, space, individual) array for the centroid; NaN where no mask
    scalar_var_arrays = {
        var: np.full((n_chunk, n_individuals), np.nan, dtype=np.float32)
        for var in SCALAR_VARS
    }
    centroid_array = np.full(
        (n_chunk, 2, n_individuals), np.nan, dtype=np.float32
    )

    # Loop thru frames in the time chunk
    frames = np.asarray(frames)
    frames = frames[(frames >= time_slice.start) & (frames < time_slice.stop)]
    for t in frames:
        props = measure_label_frame(np.asarray(labels_array[clip_index, t]))

        # Get position along `individual` of each label
        labels = props["label"]
        unknown = labels >= len(individual_of_label)
        unknown[~unknown] = individual_of_label[labels[~unknown]] < 0
        if unknown.any():
            raise ValueError(
                f"Clip index {clip_index}, frame {t}: the label image holds "
                f"labels {labels[unknown].tolist()} that are not in "
                "`label_of`."
            )
        i, k = t - time_slice.start, individual_of_label[labels]

        # Fill in arrays (values are rounded to float32 here)
        for var in SCALAR_VARS:
            scalar_var_arrays[var][i, k] = props[var]
        centroid_array[i, 0, k] = props["mask_centroid_x"]
        centroid_array[i, 1, k] = props["mask_centroid_y"]

    return xr.Dataset(
        {
            **{
                var: (("time", "individual"), arr)
                for var, arr in scalar_var_arrays.items()
            },
            "mask_centroid": (("time", "space", "individual"), centroid_array),
        }
    )


def _invert_label_of(label_of: np.ndarray) -> np.ndarray:
    """Map each label to its position along `individual`, -1 if none."""
    individual_of_label = np.full(int(label_of.max()) + 1, -1)
    individual_of_label[label_of] = np.arange(len(label_of))
    return individual_of_label


def add_mask_props_to_group(
    store_path: str | Path, video_id: str, overwrite: bool = False
) -> None:
    """Measure the masks of one masks group and write the 8 variables.

    The group must hold the copied boxes (see
    `copy_video_bboxes_to_masks`). Only frames with a finite `position`
    are measured. The 8 variables (see `compute_clip_mask_props_dataset`)
    are float32, with the chunks of the copied `confidence` (and of
    `position`, for `mask_centroid`).

    The variables are first created with no data, and then each time chunk
    of each clip is measured and written with one region write, so only one
    time chunk of values is held in memory at a time.

    Every read and write skips the root's consolidated metadata, which may
    be stale while the store is being written.

    Parameters
    ----------
    store_path : str | Path
        Path to the masks zarr store.
    video_id : str
        Name of the video group.
    overwrite : bool
        Whether to replace the 8 variables if they already exist.
        Default: False.

    Raises
    ------
    ValueError
        Before anything is written, if `label_of` has a repeated value or
        a 0 (the background label), if the group has no `position`, or if
        any of the 8 variables exists and `overwrite` is False. While
        measuring, if a label image holds a label not in `label_of`.

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
    existing = [var for var in MASK_PROPS_VARS if var in ds]
    if existing and not overwrite:
        raise ValueError(
            f"{video_id}: the group already holds {existing}. Pass "
            "overwrite=True to replace them."
        )

    # Invert label_of: label -> position along individual
    individual_of_label = _invert_label_of(label_of)

    # Get chunks of the copied boxes
    scalar_chunks = ds.confidence.encoding["chunks"]  # (clip, time, ind)
    centroid_chunks = ds.position.encoding["chunks"]  # (clip, time, sp, ind)
    n_clips, n_frames = ds.sizes["clip_id"], ds.sizes["time"]
    n_individuals, n_space = ds.sizes["individual"], ds.sizes["space"]

    # Create the 8 variables, writing metadata only
    ds_template = xr.Dataset(
        {
            **{
                var: (
                    ("clip_id", "time", "individual"),
                    da.full(
                        (n_clips, n_frames, n_individuals),
                        np.nan,
                        dtype=np.float32,
                        chunks=scalar_chunks,
                    ),
                )
                for var in SCALAR_VARS
            },
            "mask_centroid": (
                ("clip_id", "time", "space", "individual"),
                da.full(
                    (n_clips, n_frames, n_space, n_individuals),
                    np.nan,
                    dtype=np.float32,
                    chunks=centroid_chunks,
                ),
            ),
        },
        # copy the group's attributes, as to_zarr(mode="a") replaces them
        attrs=ds.attrs,
    )
    ds_template.to_zarr(
        store_path,
        group=video_id,
        mode="a",
        compute=False,
        consolidated=False,
    )

    # Get labels array and time chunks
    labels_array = zarr.open_group(
        store_path, mode="r", use_consolidated=False
    )[f"{video_id}/labels"]
    time_chunk = scalar_chunks[1]
    time_starts = range(0, n_frames, time_chunk)

    # Loop thru clips and time chunks
    for clip_index, clip_id in enumerate(ds.clip_id.values):
        start_time = time.perf_counter()
        for start in time_starts:
            time_slice = slice(start, min(start + time_chunk, n_frames))

            # Measure masks of frames with boxes
            frames = start + frames_with_boxes(
                ds[["position"]].isel(clip_id=clip_index, time=time_slice)
            )
            ds_props = compute_clip_mask_props_dataset(
                labels_array,
                clip_index,
                time_slice,
                frames,
                individual_of_label,
            )

            # Write one chunk along clip_id and time
            ds_props.expand_dims("clip_id").to_zarr(
                store_path,
                group=video_id,
                region={
                    "clip_id": slice(clip_index, clip_index + 1),
                    "time": time_slice,
                },
                consolidated=False,
            )
        logging.info(
            f"Measured masks of {video_id}/{clip_id} in "
            f"{time.perf_counter() - start_time:.1f} s"
        )
