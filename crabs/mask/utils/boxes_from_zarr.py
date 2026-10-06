"""Read tracked bounding boxes from a trajectories zarr store."""

import numpy as np
import xarray as xr


def _validate_time_axis_is_dense(
    ds: xr.Dataset, n_clip_frames: int, clip_id: str
) -> None:
    """Check there is one stored row per frame of the clip.

    This is what makes "row p is clip frame p" true, and it is the only
    property the reader relies on. Note we count finite `escape_state`
    rows rather than comparing `ds.sizes["time"]`: after the outer join
    along `clip_id`, every clip of a video reports the video's longest
    time axis.
    """
    n_frames_stored = int(np.isfinite(ds.escape_state.values).sum())
    if n_frames_stored != n_clip_frames:
        raise ValueError(
            f"{clip_id}: the time axis holds {n_frames_stored} of the "
            f"clip's {n_clip_frames} frames, so row p is not clip frame p "
            "and the masks would land on the wrong frames. This store "
            "predates PR #291; rebuild it with create-zarr-dataset before "
            "masking."
        )


def read_tracked_bboxes_from_zarr(
    ds_video: xr.Dataset, clip_id: str
) -> tuple[dict, list[str]]:
    """Read one clip of a trajectories store as tracked boxes.

    A dask-backed clip is read one `time` chunk at a time, and the boxes
    are built frame by frame, to avoid memory peaks. A long clip can hold
    thousands of track IDs  but only tens of crabs per frame, so as one dense
    (time, individual) array it is mostly NaN but can take tens of GB.

    Parameters
    ----------
    ds_video : xr.Dataset
        The dataset for one video group of a trajectories zarr store.
        It is taken already open, so the caller opens the datatree once
        for the whole run.
    clip_id : str
        The clip to read.

    Returns
    -------
    tuple[dict, list[str]]
        A map from clip frame index to
        ``{"tracked_boxes": (n, 4) float, "ids": (n,) labels}``, holding
        no key for a frame with no boxes; and the list of `individual`
        labels present in this clip.

    """
    # Get clip dataset for the range of frames in clip
    ds_clip_all = ds_video.sel(clip_id=clip_id)
    n_clip_frames = (
        int(
            ds_clip_all.clip_last_frame_0idx
            - ds_clip_all.clip_first_frame_0idx
        )
        + 1
    )
    _validate_time_axis_is_dense(ds_clip_all, n_clip_frames, clip_id)
    ds_clip = ds_clip_all.isel(time=slice(0, n_clip_frames))

    # get the clip's individuals, without the NaN padding
    in_clip = ds_clip.position.notnull().any(dim=("time", "space")).values
    all_ids = ds_clip.individual.values
    ids_in_clip = all_ids[in_clip].tolist()

    # loop one block per dask chunk along time
    time_chunks = ds_clip.position.chunksizes.get(
        "time",
        (n_clip_frames,),  # or the whole clip if not a dask array
    )
    block_ends = np.cumsum(time_chunks)

    boxes_per_frame = {}
    for start, end in zip(block_ends - time_chunks, block_ends, strict=True):
        # Get data for one block (/chunk)
        # x,y = centroid; w,h = box full width and height
        ds_block = ds_clip.isel(time=slice(start, end))
        xy, wh = ds_block.position.values, ds_block.shape.values  # (t, 2, m)

        # Loop thru frames, keep only IDs present per frame
        for i, frame_clip in enumerate(ds_block.time.values):
            present = ~np.isnan(xy[i, 0])
            if present.any():
                xy_t, wh_t = xy[i][:, present], wh[i][:, present]  # (2, n)
                box_corners = np.concatenate(
                    [xy_t - wh_t / 2, xy_t + wh_t / 2]
                )
                boxes_per_frame[int(frame_clip)] = {
                    "tracked_boxes": box_corners.T.astype(np.float64),
                    "ids": all_ids[present],
                }
    return boxes_per_frame, ids_in_clip
