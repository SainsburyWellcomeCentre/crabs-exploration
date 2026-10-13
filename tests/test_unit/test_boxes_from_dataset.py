import numpy as np
import pytest
import xarray as xr

from crabs.mask.utils.boxes_from_dataset import extract_bboxes_from_clip
from crabs.zarr.create_dataset import concat_clips


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


@pytest.fixture()
def ds_video(clip_dataset_factory) -> xr.Dataset:
    """Return a video group of two clips, of different lengths.

    Clip "Loop00" holds 2 individuals over 3 frames, with no boxes in the
    last frame. Clip "Loop01" holds 3 individuals over 4 frames.
    """
    position_0 = np.full((3, 2, 2), np.nan)
    position_0[0] = [[10.0, 30.0], [20.0, 40.0]]  # x, y per individual
    position_0[1] = [[11.0, np.nan], [21.0, np.nan]]  # id_1 absent
    shape_0 = np.full((3, 2, 2), 4.0)

    position_1 = np.zeros((4, 2, 3))
    shape_1 = np.full((4, 2, 3), 2.0)

    return concat_clips(
        [
            clip_dataset_factory("Loop00", position_0, shape_0),
            clip_dataset_factory("Loop01", position_1, shape_1),
        ]
    )


def test_extract_bboxes_from_clip(ds_video: xr.Dataset):
    """Check boxes are `position +/- shape/2`, keyed by clip frame index."""
    # Get boxes for a clip
    boxes = extract_bboxes_from_clip(ds_video.sel(clip_id="Loop00"))

    # Check only 2 frames have boxes
    assert sorted(boxes) == [0, 1]

    # Check the data
    # (expected corners are centre -/+ shape/2, with w = h = 4)
    # frame 0: centres (10, 20) and (30, 40)
    assert list(boxes[0]["ids"]) == ["id_0", "id_1"]
    np.testing.assert_allclose(
        boxes[0]["tracked_boxes"],
        [
            [8.0, 18.0, 12.0, 22.0],
            [28.0, 38.0, 32.0, 42.0],
        ],
    )

    # frame 1: only id_0 present, centred at (11, 21)
    assert list(boxes[1]["ids"]) == ["id_0"]
    np.testing.assert_allclose(
        boxes[1]["tracked_boxes"],
        [[9.0, 19.0, 13.0, 23.0]],
    )


@pytest.mark.parametrize("time_chunk", [1, 2, 3, 1000])
def test_extract_bboxes_from_clip_dask_chunks(
    ds_video: xr.Dataset, time_chunk: int
):
    """Test that the chunking of a video dataset does not change output.

    The chunking of a video dataset should not change the extracted boxes
    per frame for each clip. Chunks of 1 to 3 frames make the clips split
    across chunk boundaries. With this test we indirectly ensure the keys
    in the output are clip frame indices, not indices within a chunk.
    """
    # Compute chunked dataset
    ds_chunked = ds_video.chunk({"time": time_chunk})

    # Loop thru clips
    for clip_id in ["Loop00", "Loop01"]:
        # get data from un-chunked dataset
        unchunked_boxes = extract_bboxes_from_clip(
            ds_video.sel(clip_id=clip_id)
        )
        # get data from chunked dataset
        chunked_boxes = extract_bboxes_from_clip(
            ds_chunked.sel(clip_id=clip_id)
        )

        # check frames w data are the same in chunked and unchunked output
        # (Note: sorted returns list of sorted keys)
        assert sorted(chunked_boxes) == sorted(unchunked_boxes)

        # compare boxes per (unchunked) frame
        for frame, unchunked_boxes_one_frame in unchunked_boxes.items():
            np.testing.assert_array_equal(
                chunked_boxes[frame]["tracked_boxes"],
                unchunked_boxes_one_frame["tracked_boxes"],
            )
            assert list(chunked_boxes[frame]["ids"]) == list(
                unchunked_boxes_one_frame["ids"]
            )


def test_extract_bboxes_from_clip_rejects_sparse_time_axis(
    clip_dataset_factory,
):
    """Test that a clip with frames missing from its time axis throws error.

    Stores built before PR #291 only hold frames with tracked boxes,
    so a clip could have fewer `time` rows (i.e. frames with data) than
    frames. Here "Loop00" spans 6 frames but has data for only 4.

    Checking `sizes["time"]` would not catch this: concatenating clips
    pads every clip to the longest one ("Loop01", 6 frames), with NaNs.
    The check instead counts the rows with a finite `escape_state`,
    which excludes that padding.
    """
    # Define a video dataset with a sparse clip
    ds_video = concat_clips(
        [
            # clip_id = 0
            # 4 stored rows for a 6-frame clip
            clip_dataset_factory(
                "Loop00",
                np.zeros((4, 2, 1)),  # position
                np.ones((4, 2, 1)),  # shape
                n_clip_frames=6,
            ),
            # clip_id = 1
            clip_dataset_factory(
                "Loop01",
                np.zeros((6, 2, 1)),
                np.ones((6, 2, 1)),
            ),
        ]
    )

    # The length of the clip_id time axis is 6 frames
    assert ds_video.sel(clip_id="Loop00").sizes["time"] == 6

    with pytest.raises(ValueError, match="Loop00: the time axis holds 4"):
        extract_bboxes_from_clip(ds_video.sel(clip_id="Loop00"))


def test_extract_bboxes_from_clip_rejects_video_dataset(
    ds_video: xr.Dataset,
):
    """Test that passing a whole video group, not a clip, throws error."""
    with pytest.raises(ValueError, match="Expected a single clip"):
        extract_bboxes_from_clip(ds_video)
