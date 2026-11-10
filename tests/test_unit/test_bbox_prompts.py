from pathlib import Path

import numpy as np
import pytest
import xarray as xr

from crabs.mask.mask_video import create_mask_store
from crabs.mask.utils.bbox_prompts import (
    BBOX_VARS,
    copy_video_bboxes_to_masks,
    extract_bboxes_from_clip,
)
from crabs.zarr.create_dataset import concat_clips


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
            # 4 stored rows for a 6-frame clip
            clip_dataset_factory(
                "Loop00",
                np.zeros((4, 2, 1)),  # position
                np.ones((4, 2, 1)),  # shape
                n_clip_frames=6,
            ),
            clip_dataset_factory(
                "Loop01",
                np.zeros((6, 2, 1)),
                np.ones((6, 2, 1)),
            ),
        ]
    )

    # Loop00 is padded to Loop01's 6 frames by the outer join
    assert ds_video.sel(clip_id="Loop00").sizes["time"] == 6

    with pytest.raises(ValueError, match="Loop00: the time axis holds 4"):
        extract_bboxes_from_clip(ds_video.sel(clip_id="Loop00"))


def test_extract_bboxes_from_clip_rejects_video_dataset(
    ds_video: xr.Dataset,
):
    """Test that passing a whole video group, not a clip, throws error."""
    with pytest.raises(ValueError, match="Expected a single clip"):
        extract_bboxes_from_clip(ds_video)


MASKS_ATTRS = {"occlusion_policy": "smallest_wins", "boxes": {"file": "t"}}


def write_trajectories_group(
    ds_video: xr.Dataset, store_path: Path, time_chunk: int = 2
) -> xr.Dataset:
    """Write `ds_video` as float32 to a trajectories store and read it back.

    The box variables are chunked along `time`, so their chunks are
    smaller than a clip, as in real stores.
    """
    ds_video = ds_video.copy()
    for var in BBOX_VARS:
        ds_video[var] = ds_video[var].astype(np.float32)
    ds_video.attrs["video_id"] = "video"
    ds_video.chunk({"time": time_chunk}).to_zarr(store_path, group="video")
    return xr.open_zarr(store_path, group="video")


def create_masks_group(
    store_path: Path,
    n_frames: int,
    clip_ids: list[str] | None = None,
    individuals: list[str] | None = None,
) -> None:
    """Create an old masks group: labels, label_of and attrs only."""
    create_mask_store(
        store_path=store_path,
        video_id="video",
        clip_ids=clip_ids or ["Loop00", "Loop01"],
        n_frames=n_frames,
        individuals=individuals or ["id_0", "id_1", "id_2"],
        image_shape=(8, 8),
        metadata_dict=MASKS_ATTRS,
        n_frames_per_shard=None,
        zarr_mode_group="w-",
    )


def open_masks_group(store_path: Path) -> xr.Dataset:
    """Open the masks group as it is on disk."""
    return xr.open_zarr(store_path, group="video", consolidated=False)


def test_copy_video_bboxes_to_masks(ds_video: xr.Dataset, tmp_path: Path):
    """Check the copied boxes equal the source, with its chunks and dtype.

    The masks group's own arrays and attributes must be unchanged.
    """
    # Write trajectories and old masks group
    ds_tracks = write_trajectories_group(ds_video, tmp_path / "tracks.zarr")
    masks_path = tmp_path / "masks.zarr"
    create_masks_group(masks_path, n_frames=ds_tracks.sizes["time"])
    ds_masks_before = open_masks_group(masks_path).load()

    # Copy boxes
    copy_video_bboxes_to_masks(masks_path, ds_tracks)

    # Check boxes equal the source
    ds_masks = open_masks_group(masks_path)
    for var in BBOX_VARS:
        xr.testing.assert_equal(
            ds_masks[var].reset_coords(drop=True),
            ds_tracks[var].reset_coords(drop=True),
        )
        assert ds_masks[var].dtype == np.float32
        assert (
            ds_masks[var].encoding["chunks"]
            == (ds_tracks[var].encoding["chunks"])
        )

    # Check no clip coordinates were copied into the masks group
    assert "clip_first_frame_0idx" not in ds_masks.coords

    # Check the masks group's own data and attrs are unchanged
    xr.testing.assert_identical(
        ds_masks[["labels", "label_of"]].load(),
        ds_masks_before[["labels", "label_of"]],
    )
    assert ds_masks.attrs == MASKS_ATTRS


def test_copy_video_bboxes_to_masks_pads_time(
    ds_video: xr.Dataset, tmp_path: Path
):
    """Check boxes are padded with NaN when the masks `time` is longer."""
    # Write trajectories store with 4 frames, and masks store with 6
    ds_tracks = write_trajectories_group(ds_video, tmp_path / "tracks.zarr")
    masks_path = tmp_path / "masks.zarr"
    create_masks_group(masks_path, n_frames=6)

    # Copy boxes
    copy_video_bboxes_to_masks(masks_path, ds_tracks)

    # Check
    ds_masks = open_masks_group(masks_path)
    assert ds_masks.position.sizes["time"] == 6
    np.testing.assert_array_equal(
        ds_masks.position.isel(time=slice(0, 4)).values,
        ds_tracks.position.values,
    )
    assert ds_masks.position.isel(time=slice(4, None)).isnull().all()
    assert ds_masks.confidence.isel(time=slice(4, None)).isnull().all()


def test_copy_video_bboxes_to_masks_cuts_time(
    ds_video: xr.Dataset, tmp_path: Path
):
    """Check boxes are cut when the masks `time` is shorter, with no box lost.

    The last frame of the trajectories store has no box, so it can be
    dropped.
    """
    # Remove boxes in last frame (time=3)
    no_box_at_end = ds_video.time < 3
    for var in BBOX_VARS:
        ds_video[var] = ds_video[var].where(no_box_at_end)
    ds_tracks = write_trajectories_group(ds_video, tmp_path / "tracks.zarr")

    # Initialise a masks store with 3 frames only
    masks_path = tmp_path / "masks.zarr"
    create_masks_group(masks_path, n_frames=3)

    # Copy boxes
    copy_video_bboxes_to_masks(masks_path, ds_tracks)

    # Check
    ds_masks = open_masks_group(masks_path)
    assert ds_masks.position.sizes["time"] == 3
    np.testing.assert_array_equal(
        ds_masks.position.values,
        ds_tracks.position.isel(time=slice(0, 3)).values,
    )


@pytest.mark.parametrize(
    ("masks_kwargs", "sparse_time", "expected_message"),
    [
        ({"n_frames": 3}, False, "past the end"),
        (
            {"n_frames": 4, "clip_ids": ["Loop00", "Loop02"]},
            False,
            "`clip_id`",
        ),
        (
            {"n_frames": 4, "individuals": ["id_0", "id_1", "id_9"]},
            False,
            "`individual`",
        ),
        ({"n_frames": 8}, True, "not 0, 1, ..., n - 1"),
    ],
)
def test_copy_video_bboxes_to_masks_errors(
    ds_video: xr.Dataset,
    tmp_path: Path,
    masks_kwargs: dict,
    sparse_time: bool,
    expected_message: str,
):
    """Check errors are raised before anything is written.

    The cases are: a box past the end of the masks `time`, a mismatched
    `clip_id` or `individual`, and a trajectories `time` that is not the
    frame index (as in stores built before PR #291).
    """
    if sparse_time:
        ds_video = ds_video.assign_coords(time=ds_video.time * 2)
    ds_tracks = write_trajectories_group(ds_video, tmp_path / "tracks.zarr")
    masks_path = tmp_path / "masks.zarr"
    create_masks_group(masks_path, **masks_kwargs)

    with pytest.raises(ValueError, match=expected_message):
        copy_video_bboxes_to_masks(masks_path, ds_tracks)

    # Check nothing was written
    ds_masks = open_masks_group(masks_path)
    assert not set(BBOX_VARS) & set(ds_masks.data_vars)
    assert ds_masks.attrs == MASKS_ATTRS
