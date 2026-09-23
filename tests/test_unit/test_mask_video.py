from pathlib import Path

import cv2
import numpy as np
import pytest
import xarray as xr
import zarr

from crabs.tracker.mask_video import (
    create_mask_store,
    write_clip_masks_to_store,
)

FRAME_SHAPE = (64, 64)  # height, width
N_FRAMES = 5
SHARD_N_FRAMES = 2  # two whole shards plus a one-frame tail
INDIVIDUALS = ["id_0", "id_1", "id_2"]
ATTRS = {
    "occlusion_policy": "smallest_wins",
    "background_label": 0,
    "boxes": {"file": "tracks.zarr", "source": "trajectories_zarr"},
}


class FakePredictor:
    """A predictor returning each box's own rectangle as its mask.

    It reproduces the squeeze in `SAM2ImagePredictor.predict`, which
    returns (N, 1, H, W) for N>1 but (1, H, W) for N==1.
    """

    def __init__(self):
        """Initialise the count of frames the predictor was shown."""
        self.n_frames_seen = 0

    def set_image(self, image: np.ndarray) -> None:
        """Count the frame; its pixels are ignored."""
        self.n_frames_seen += 1

    def predict(self, box: np.ndarray, multimask_output: bool):
        """Return one full-frame mask per box."""
        masks = np.zeros((len(box), *FRAME_SHAPE), dtype=np.float32)
        for mask, (x1, y1, x2, y2) in zip(masks, box.astype(int), strict=True):
            mask[y1:y2, x1:x2] = 1.0
        if len(box) == 1:
            return masks, None, None
        return masks[:, np.newaxis], None, None


@pytest.fixture()
def clip_video(tmp_path: Path) -> str:
    """Write a blank clip video; the fake predictor ignores its pixels."""
    path = str(tmp_path / "video-Loop00.mp4")
    writer = cv2.VideoWriter(
        path, cv2.VideoWriter_fourcc(*"mp4v"), 10, FRAME_SHAPE[::-1]
    )
    for _ in range(N_FRAMES):
        writer.write(np.zeros((*FRAME_SHAPE, 3), dtype=np.uint8))
    writer.release()
    return path


@pytest.fixture()
def mask_store(tmp_path: Path) -> tuple[Path, zarr.Array]:
    """Return the path of a two-clip mask store and its labels array."""
    store_path = tmp_path / "masks.zarr"
    labels_array = create_mask_store(
        store_path=store_path,
        video_id="video",
        clip_ids=["Loop00", "Loop01"],
        n_frames=N_FRAMES,
        individuals=INDIVIDUALS,
        image_shape=FRAME_SHAPE,
        metadata_dict=ATTRS,
        shard_n_frames=SHARD_N_FRAMES,
        zarr_mode_group="w-",
    )
    return store_path, labels_array


def test_create_mask_store(mask_store: tuple[Path, zarr.Array]):
    """The template is on disk, with no data written yet."""
    store_path, labels_array = mask_store
    height, width = FRAME_SHAPE

    assert labels_array.shape == (2, N_FRAMES, height, width)
    assert labels_array.dtype == np.uint16
    assert labels_array.chunks == (1, 1, height, width)
    assert labels_array.shards == (1, SHARD_N_FRAMES, height, width)

    # a template write puts metadata and coordinates on disk, but no
    # labels: each clip fills its own region afterwards
    assert (store_path / "video/labels/zarr.json").is_file()
    assert not (store_path / "video/labels/c").exists()

    ds = xr.open_datatree(store_path, engine="zarr")["video"].to_dataset()
    assert ds.labels.dims == ("clip_id", "time", "img_h", "img_w")
    assert list(ds.clip_id.values) == ["Loop00", "Loop01"]
    assert list(ds.individual.values) == INDIVIDUALS
    # the mapping is explicit, so no reader needs to know the offset
    assert ds.label_of.dims == ("individual",)
    assert list(ds.label_of.values) == [1, 2, 3]
    # a nested attr survives the round trip through zarr's JSON attributes
    assert ds.attrs == ATTRS


@pytest.mark.parametrize("max_prompts_per_batch", [1, 2, 8])
def test_write_clip_masks_to_store(
    mask_store: tuple[Path, zarr.Array],
    clip_video: str,
    max_prompts_per_batch: int,
):
    """One clip pass, painted by label and flushed shard by shard.

    The written frames must not depend on `max_prompts_per_batch`: it is
    documented as a memory knob, so it must not change the pixels.
    """
    store_path, labels_array = mask_store
    label_of = xr.open_datatree(store_path, engine="zarr")["video"].label_of

    tracked_bboxes_dict = {
        # the smaller crab is given first, so the pre-sort by box area is
        # what puts the larger one in the earlier prompt chunk
        0: {
            "tracked_boxes": np.array(
                [[24.0, 24.0, 40.0, 40.0], [0.0, 0.0, 32.0, 32.0]]
            ),
            "ids": np.array(["id_2", "id_0"], dtype=object),
            "scores": np.array([0.9, 0.8]),  # an extra key is ignored
        },
        # frame 1 has no key at all, and frame 2 a key with no boxes
        2: {
            "tracked_boxes": np.zeros((0, 4)),
            "ids": np.array([], dtype=object),
        },
        3: {
            "tracked_boxes": np.array([[40.0, 40.0, 56.0, 56.0]]),
            "ids": np.array(["id_1"], dtype=object),
        },
        4: {
            "tracked_boxes": np.array([[0.0, 0.0, 32.0, 32.0]]),
            "ids": np.array(["id_0"], dtype=object),
        },
    }

    predictor = FakePredictor()
    n_frames_read = write_clip_masks_to_store(
        video_path=clip_video,
        tracked_bboxes_dict=tracked_bboxes_dict,
        labels_array=labels_array,
        clip_index=0,
        label_of=label_of,
        predictor=predictor,
        max_prompts_per_batch=max_prompts_per_batch,
        shard_n_frames=SHARD_N_FRAMES,
    )

    assert n_frames_read == N_FRAMES
    # SAM2 is skipped entirely on a frame with no boxes
    assert predictor.n_frames_seen == 3

    labels = np.asarray(labels_array[0])

    # frame 0: smallest_wins, so id_2 owns the contested pixels
    expected = np.zeros(FRAME_SHAPE, dtype=np.uint16)
    expected[0:32, 0:32] = 1  # id_0
    expected[24:40, 24:40] = 3  # id_2, painted last
    np.testing.assert_array_equal(labels[0], expected)

    # frames with no boxes are all background, not skipped
    assert not labels[1].any()
    assert not labels[2].any()

    # frame 3 is the single-prompt case, where predict() squeezes
    assert set(np.unique(labels[3])) == {0, 2}
    assert labels[3, 40:56, 40:56].min() == 2

    # frame 4 is the part-filled shard tail, easy to drop silently
    assert labels[4, 0:32, 0:32].min() == 1

    # the other clip is untouched by this one's writes
    assert not np.asarray(labels_array[1]).any()
