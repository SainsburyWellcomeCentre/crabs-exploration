"""Utils to support the generation of masks zarr stores."""

from pathlib import Path

import numpy as np
import xarray as xr

# Variables of bboxes store to copy to mask store
BBOX_VARS = ["position", "shape", "confidence"]


def extract_bboxes_from_clip(ds_clip: xr.Dataset) -> dict:
    """Extract boxes from a clip dataset as a dictionary.

    Parameters
    ----------
    ds_clip : xr.Dataset
        The trajectory data for one clip. Usually selected from a video group
        of a trajectories zarr store with ``ds_video.sel(clip_id=...)``.

    Returns
    -------
    dict
        A dictionary mapping clip frame index to bounding boxes coordinates
        and IDs. Coordinates are in corner format ``(x_min, y_min, x_max,
        y_max)`` as an (n,4) arrays. Only frames
        with bounding boxes are included in the output.

    """
    if "clip_id" in ds_clip.dims:
        raise ValueError(
            "Expected a single clip; select one with .sel(clip_id=...)."
        )

    # Validate time axis
    clip_id = str(ds_clip.clip_id.item())
    n_frames_in_clip = (
        int(ds_clip.clip_last_frame_0idx - ds_clip.clip_first_frame_0idx) + 1
    )
    _validate_time_axis_is_dense(ds_clip, clip_id, n_frames_in_clip)

    # Drop padding frames
    ds_clip = ds_clip.isel(time=slice(0, n_frames_in_clip))

    # Get list of individuals
    # (includes padding to max individuals in video)
    individuals_in_clip = ds_clip.individual.values

    # Compute start and end of each time chunk
    time_chunks = ds_clip.position.chunksizes.get(
        "time",
        (n_frames_in_clip,),  # or the whole clip if not a dask array
    )
    chunk_ends = np.cumsum(time_chunks)
    chunk_starts = chunk_ends - time_chunks

    # Loop thru one dask chunk at a time
    boxes_per_frame = {}
    for start, end in zip(chunk_starts, chunk_ends, strict=True):
        # Get all box data for one chunk
        ds_block = ds_clip.isel(time=slice(start, end))
        boxes_xy, boxes_wh = (
            ds_block.position.values,
            ds_block.shape.values,
        )  # (t, 2, m)

        # Loop thru frames, extract boxes of only IDs present per frame
        for i, frame_clip in enumerate(ds_block.time.values):
            present = ~np.isnan(boxes_xy[i, 0])
            if present.any():
                xy_t, wh_t = (
                    boxes_xy[i][:, present],
                    boxes_wh[i][:, present],
                )  # (2, n)
                box_corners = np.concatenate(
                    [xy_t - wh_t / 2, xy_t + wh_t / 2]
                )
                boxes_per_frame[int(frame_clip)] = {
                    "tracked_boxes": box_corners.T.astype(np.float64),
                    "ids": individuals_in_clip[present],
                }
    return boxes_per_frame


def _validate_time_axis_is_dense(
    ds: xr.Dataset,
    clip_id: str,
    n_frames_in_clip: int,
) -> None:
    """Check there is one coordinate per frame of the clip.

    The input `n_frames_in_clip` is computed from `clip_first_frame_0idx`
    and `clip_last_frame_0idx`. Note that here we count finite `escape_state`
    rows rather than comparing `ds.sizes["time"]`: after the outer join
    along `clip_id`, every clip of a video reports the video's longest time
    axis.
    """
    n_frames_in_store = int(np.isfinite(ds.escape_state.values).sum())
    if n_frames_in_store != n_frames_in_clip:
        raise ValueError(
            f"{clip_id}: the time axis holds {n_frames_in_store} of the "
            f"clip's {n_frames_in_clip} frames. This store "
            "predates PR #291; rebuild it with create-zarr-dataset before "
            "predicting masks."
        )


def copy_video_bboxes_to_masks(
    mask_store_path: str | Path,
    ds_video: xr.Dataset,
) -> None:
    """Copy one video's boxes into the masks store.

    The boxes (`position`, `shape` and `confidence`) are the prompts
    used to predict the masks.

    Parameters
    ----------
    mask_store_path : str | Path
        Path to the masks zarr store.
    ds_video : xr.Dataset
        The video group of the trajectories store, as read from zarr.

    Raises
    ------
    ValueError
        If `clip_id` or `individual` axes differ
        between the two groups, if the trajectories `time` is not
        ``0, 1, ..., n - 1``, or if a box falls past the end of the masks
        group's `time`.

    """
    # Read masks group
    video_id = ds_video.attrs["video_id"]
    ds_masks = xr.open_zarr(
        mask_store_path,
        group=video_id,
        consolidated=False,
    )

    # Check both groups share clip and individual coordinates
    for coord in ["clip_id", "individual"]:
        if not np.array_equal(
            ds_video[coord].values.astype(str),
            ds_masks[coord].values.astype(str),
        ):
            raise ValueError(
                f"{video_id}: the `{coord}` coordinate of the trajectories "
                "store differs from that of the masks store."
            )

    # Check the trajectories time axis is the clip frame index
    if not np.array_equal(
        ds_video.time.values,
        np.arange(ds_video.sizes["time"]),
    ):
        raise ValueError(
            f"{video_id}: the `time` coordinate of the trajectories store "
            "is not 0, 1, ..., n - 1. This store predates PR #291; rebuild "
            "it with create-zarr-dataset."
        )

    # Check no boxes exist past the masks time end
    n_frames_masks = ds_masks.sizes["time"]
    if bool(
        ds_video.position.isel(time=slice(n_frames_masks, None))
        .notnull()
        .any()
    ):
        raise ValueError(
            f"{video_id}: the trajectories store has boxes at time "
            f">= {n_frames_masks}, past the end of the masks group's time "
            "axis."
        )

    # Define bbox dataset to append
    ds_boxes_to_append = ds_video[BBOX_VARS].reindex(time=ds_masks.time.values)

    # Reindexing alters the dask chunks, so restore the trajectories' chunks
    chunks = dict(
        zip(
            ds_video.confidence.dims,
            ds_video.confidence.encoding["chunks"],
            strict=True,
        )
    )
    ds_boxes_to_append = (
        ds_boxes_to_append.drop_encoding()
        .chunk(chunks)
        # drop all coordinates to not overwrite the masks' ones,
        # except `space`, which does not exist in the mask store
        # (coords should be the same but they may vary in dtype etc)
        .drop_vars([c for c in ds_boxes_to_append.coords if c != "space"])
    )

    # to_zarr(mode="a") replaces the group's attrs with the dataset's;
    # we pass the masks group's own attrs to keep them unchanged
    ds_boxes_to_append.attrs = ds_masks.attrs

    # Append to masks zarr
    ds_boxes_to_append.to_zarr(
        mask_store_path,
        group=video_id,
        mode="a",
        consolidated=False,
    )
