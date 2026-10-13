"""Pytest fixtures for trajectories datasets."""

import numpy as np
import pytest
import xarray as xr


@pytest.fixture()
def clip_dataset_factory():
    """Return a factory of single-clip trajectories datasets."""

    def _clip_dataset(
        clip_id: str,
        position: np.ndarray,
        shape: np.ndarray,
        n_clip_frames: int | None = None,
    ) -> xr.Dataset:
        """Build one clip of a trajectories store.

        `position` and `shape` are (time, space, individual).
        `n_clip_frames` defaults to the number of stored rows; passing a
        larger value mimics a pre-291 store, whose time axis is shorter
        than the clip.
        """
        n_stored, _, n_individuals = position.shape
        n_clip_frames = n_clip_frames or n_stored
        ds = xr.Dataset(
            {
                "position": (("time", "space", "individual"), position),
                "shape": (("time", "space", "individual"), shape),
                "escape_state": (
                    ("time",),
                    np.zeros(n_stored, dtype=np.float16),
                ),
            },
            coords={
                "time": np.arange(n_stored),
                "space": ["x", "y"],
                "individual": [f"id_{i}" for i in range(n_individuals)],
            },
        )
        return ds.expand_dims("clip_id").assign_coords(
            clip_id=np.array([clip_id], dtype=str),
            clip_first_frame_0idx=("clip_id", [0]),
            clip_last_frame_0idx=("clip_id", [n_clip_frames - 1]),
        )

    return _clip_dataset
