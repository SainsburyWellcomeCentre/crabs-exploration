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
    ds = ds_video.sel(clip_id=clip_id)
    n_clip_frames = int(ds.clip_last_frame_0idx - ds.clip_first_frame_0idx) + 1

    _validate_time_axis_is_dense(ds, n_clip_frames, clip_id)

    ds = ds.isel(time=slice(0, n_clip_frames))
    # drop the NaN padding added when concatenating clips along clip_id
    ds = ds.dropna(dim="individual", how="all")

    # movement stores the centroid and the box size
    xy, wh = ds.position.values, ds.shape.values  # (t, 2, m)
    corners = np.concatenate([xy - wh / 2, xy + wh / 2], axis=1)  # (t, 4, m)

    individuals = [str(v) for v in ds.individual.values]
    boxes = {}
    for i, t in enumerate(ds.time.values):
        present = ~np.isnan(corners[i, 0])
        if present.any():
            boxes[int(t)] = {
                "tracked_boxes": corners[i][:, present].T.astype(np.float64),
                "ids": np.array(individuals, dtype=object)[present],
            }
    return boxes, individuals
