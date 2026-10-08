"""Read tracked bounding boxes from a trajectories zarr store."""

import numpy as np
import xarray as xr


def read_tracked_bboxes_from_zarr(
    ds_video: xr.Dataset,
    clip_id: str,
) -> dict:
    """Extract boxes from a single clip.

    Parameters
    ----------
    ds_video : xr.Dataset
        The trajectory data for one video group of a trajectories zarr store.
    clip_id : str
        The clip to extract the boxes from.

    Returns
    -------
    dict
        A dictionary mapping clip frame index to bounding boxes coordinates
        and IDs, formatted as:
        ``{"tracked_boxes": (n, 4) float, "ids": (n,) labels}``. Only frames
        with bounding boxes are included in the output.

    """
    # Get clip dataset
    ds_clip_all = ds_video.sel(clip_id=clip_id)
    n_frames_in_clip = (
        int(
            ds_clip_all.clip_last_frame_0idx
            - ds_clip_all.clip_first_frame_0idx
        )
        + 1
    )
    _validate_time_axis_is_dense(ds_clip_all, clip_id, n_frames_in_clip)

    # Drop padding frames
    ds_clip = ds_clip_all.isel(time=slice(0, n_frames_in_clip))

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
