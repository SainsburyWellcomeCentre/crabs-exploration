import numpy as np
import pytest
import xarray as xr

from crabs.mask.utils.boxes_from_zarr import read_tracked_bboxes_from_zarr


def clip_dataset(
    clip_id: str,
    position: np.ndarray,
    shape: np.ndarray,
    n_clip_frames: int | None = None,
) -> xr.Dataset:
    """Build one clip of a trajectories store.

    `position` and `shape` are (time, space, individual). `n_clip_frames`
    defaults to the number of stored rows; passing a larger value mimics a
    pre-291 store, whose time axis is shorter than the clip.
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


def concat_clips(list_ds: list[xr.Dataset]) -> xr.Dataset:
    """Concatenate clip datasets as `create-zarr-dataset` does."""
    return xr.concat(
        list_ds,
        dim="clip_id",
        join="outer",
        coords="different",
        compat="equals",
    )


@pytest.fixture()
def ds_video() -> xr.Dataset:
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
            clip_dataset("Loop00", position_0, shape_0),
            clip_dataset("Loop01", position_1, shape_1),
        ]
    )


def test_read_tracked_bboxes_from_zarr(ds_video: xr.Dataset):
    """Boxes are `position +/- shape/2`, keyed by clip frame index."""
    boxes, individuals = read_tracked_bboxes_from_zarr(ds_video, "Loop00")

    # the clip's own individuals, without the padding from the longer clip
    assert individuals == ["id_0", "id_1"]

    # no key for a frame with no boxes
    assert sorted(boxes) == [0, 1]

    np.testing.assert_allclose(
        boxes[0]["tracked_boxes"],
        [[8.0, 18.0, 12.0, 22.0], [28.0, 38.0, 32.0, 42.0]],
    )
    assert list(boxes[0]["ids"]) == ["id_0", "id_1"]

    # only the individuals present in that frame
    assert list(boxes[1]["ids"]) == ["id_0"]
    np.testing.assert_allclose(
        boxes[1]["tracked_boxes"], [[9.0, 19.0, 13.0, 23.0]]
    )


@pytest.mark.parametrize("time_chunk", [1, 2, 3, 1000])
def test_read_tracked_bboxes_from_zarr_dask_chunks(
    ds_video: xr.Dataset, time_chunk: int
):
    """Test that the chunking of a video dataset does not change output.

    The chunking of a video dataset should not change the extracted boxes
    per frame for each clip. Chunks of 1 to 3 frames make the clips split
    across chunk boundaries. With this test we indirectly ensure the keys
    in the output are clip frame indices, not indices within a chunk.
    """
    ds_chunked = ds_video.chunk({"time": time_chunk})
    for clip_id in ["Loop00", "Loop01"]:
        # get data from un-chunked dataset
        unchunked_boxes, unchunked_individuals = read_tracked_bboxes_from_zarr(
            ds_video, clip_id
        )
        # get data from chunked dataset
        chunked_boxes, chunked_individuals = read_tracked_bboxes_from_zarr(
            ds_chunked, clip_id
        )

        # check individuals are the same in chunked and unchunked output
        assert chunked_individuals == unchunked_individuals

        # check frames w data are the same in chunked and unchunked output
        # (Note: sorted returns list of sorted keys)
        assert sorted(chunked_boxes) == sorted(unchunked_boxes)

        # compare boxes per unchunked frame
        for frame, unchunked_boxes_one_frame in unchunked_boxes.items():
            np.testing.assert_array_equal(
                chunked_boxes[frame]["tracked_boxes"],
                unchunked_boxes_one_frame["tracked_boxes"],
            )
            assert list(chunked_boxes[frame]["ids"]) == list(
                unchunked_boxes_one_frame["ids"]
            )


def test_read_tracked_bboxes_from_zarr_rejects_sparse_time_axis():
    """A store whose time axis is shorter than the clip is refused.

    The guard must count finite `escape_state` rows: after the outer join
    both clips report the same `sizes["time"]`, so a length comparison
    would never fire.
    """
    ds_video = concat_clips(
        [
            # 4 stored rows for a 6-frame clip
            clip_dataset(
                "Loop00",
                np.zeros((4, 2, 1)),
                np.ones((4, 2, 1)),
                n_clip_frames=6,
            ),
            clip_dataset("Loop01", np.zeros((6, 2, 1)), np.ones((6, 2, 1))),
        ]
    )
    assert ds_video.sel(clip_id="Loop00").sizes["time"] == 6

    with pytest.raises(ValueError, match="Loop00: the time axis holds 4"):
        read_tracked_bboxes_from_zarr(ds_video, "Loop00")
