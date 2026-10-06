import shutil
from pathlib import Path

import cv2
import numpy as np
import pytest
import torch
import xarray as xr
import zarr
from zarr.codecs import ZstdCodec

from crabs.mask.mask_video import (
    create_mask_store,
    main,
    mask_parse_args,
    predict_masks_on_device,
    write_clip_masks_to_store,
)
from crabs.mask.utils.boxes_from_zarr import read_tracked_bboxes_from_zarr
from tests.test_unit.test_boxes_from_zarr import clip_dataset, concat_clips

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

    It implements the two private `SAM2ImagePredictor` methods that
    `predict_masks_on_device` calls, with the same return shapes: `_predict`
    returns (N, 1, H, W) boolean masks, for any N.
    """

    device = torch.device("cpu")

    def __init__(self):
        """Initialise the count of frames the predictor was shown."""
        self.n_frames_seen = 0

    def set_image(self, image: np.ndarray) -> None:
        """Count the frame; its pixels are ignored."""
        self.n_frames_seen += 1

    def _prep_prompts(
        self, point_coords, point_labels, box, mask_logits, normalize_coords
    ):
        """Pass the boxes through untransformed, as a tensor."""
        return None, None, None, torch.as_tensor(box)

    def _predict(
        self, point_coords, point_labels, boxes, mask_input, multimask_output
    ):
        """Return one full-frame mask per box."""
        masks = torch.zeros((len(boxes), 1, *FRAME_SHAPE), dtype=torch.bool)
        for mask, (x1, y1, x2, y2) in zip(
            masks, boxes.int().tolist(), strict=True
        ):
            mask[0, y1:y2, x1:x2] = True
        return masks, None, None


@pytest.mark.parametrize("n_boxes", [1, 3])
def test_predict_masks_on_device_matches_sam2_predict(n_boxes: int):
    """The private-API path gives the masks SAM2's public `predict` does.

    It runs a randomly initialised SAM2, so no checkpoint is downloaded:
    the masks are meaningless, but must be identical. A SAM2 bump that
    changes the private methods fails here, not on the cluster.
    """
    pytest.importorskip("sam2")
    from sam2.build_sam import build_sam2
    from sam2.sam2_image_predictor import SAM2ImagePredictor

    torch.manual_seed(0)
    predictor = SAM2ImagePredictor(
        build_sam2(
            "configs/sam2.1/sam2.1_hiera_t.yaml", ckpt_path=None, device="cpu"
        )
    )
    rng = np.random.default_rng(0)
    predictor.set_image(rng.integers(0, 255, (48, 80, 3), dtype=np.uint8))
    boxes = np.array(
        [
            [2.0, 3.0, 30.0, 40.0],
            [10.0, 5.0, 70.0, 45.0],
            [40.0, 20.0, 60.0, 30.0],
        ]
    )[:n_boxes]

    masks_public, _, _ = predictor.predict(box=boxes, multimask_output=False)
    masks_on_device = predict_masks_on_device(predictor, boxes)

    assert masks_on_device.shape == (n_boxes, 48, 80)
    np.testing.assert_array_equal(
        masks_on_device.numpy(),
        masks_public.reshape(n_boxes, 48, 80).astype(bool),
    )


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
    # zarr's default compressor
    assert [type(c) for c in labels_array.compressors] == [ZstdCodec]

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
        # what puts the larger one in the earlier batch of prompts
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
        video_clip_path=clip_video,
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

    # frame 3 is the single-prompt case, which SAM2's public predict()
    # would have squeezed
    assert set(np.unique(labels[3])) == {0, 2}
    assert labels[3, 40:56, 40:56].min() == 2

    # frame 4 is the part-filled shard tail, easy to drop silently
    assert labels[4, 0:32, 0:32].min() == 1

    # the other clip is untouched by this one's writes
    assert not np.asarray(labels_array[1]).any()


def test_write_clip_masks_to_store_clip_shorter_than_store(
    tmp_path: Path, clip_video: str, caplog: pytest.LogCaptureFixture
):
    """A clip shorter than its group's longest clip is read in full.

    The store's time axis is the longest clip in the group, so reaching
    the end of a shorter clip must not be logged as a frame reading error.
    """
    store_path = tmp_path / "masks.zarr"
    labels_array = create_mask_store(
        store_path=store_path,
        video_id="video",
        clip_ids=["Loop00"],
        n_frames=N_FRAMES + 3,
        individuals=INDIVIDUALS,
        image_shape=FRAME_SHAPE,
        metadata_dict=ATTRS,
        shard_n_frames=SHARD_N_FRAMES,
        zarr_mode_group="w-",
    )
    label_of = xr.open_datatree(store_path, engine="zarr")["video"].label_of

    with caplog.at_level("INFO"):
        n_frames_read = write_clip_masks_to_store(
            video_clip_path=clip_video,
            tracked_bboxes_dict={},
            labels_array=labels_array,
            clip_index=0,
            label_of=label_of,
            predictor=FakePredictor(),
            max_prompts_per_batch=2,
            shard_n_frames=SHARD_N_FRAMES,
        )

    assert n_frames_read == N_FRAMES
    assert f"All {N_FRAMES} frames processed" in caplog.text
    assert "Error reading frame" not in caplog.text


@pytest.fixture()
def trajectories_store(tmp_path: Path) -> Path:
    """Write a one-video trajectories store and the clip videos it names.

    Clip "Loop00" runs 3 frames and holds 2 individuals, the second of
    which is absent from the last frame. Clip "Loop01" runs 2 frames and
    holds 1 individual, so it is NaN-padded along `individual`.
    """
    position_0 = np.full((3, 2, 2), np.nan)
    position_0[:2] = [[10.0, 40.0], [10.0, 40.0]]  # x, y per individual
    position_0[2] = [[10.0, np.nan], [10.0, np.nan]]  # id_1 absent
    position_1 = np.full((2, 2, 1), 20.0)

    ds_video = concat_clips(
        [
            clip_dataset("Loop00", position_0, np.full((3, 2, 2), 8.0)),
            clip_dataset("Loop01", position_1, np.full((2, 2, 1), 8.0)),
        ]
    )
    store_path = tmp_path / "CrabTracks.zarr"
    ds_video.to_zarr(store_path, group="video")

    videos_dir = tmp_path / "clips"
    videos_dir.mkdir()
    for clip_id, n_frames in [("Loop00", 3), ("Loop01", 2)]:
        writer = cv2.VideoWriter(
            str(videos_dir / f"video-{clip_id}.mp4"),
            cv2.VideoWriter_fourcc(*"mp4v"),
            10,
            FRAME_SHAPE[::-1],
        )
        for _ in range(n_frames):
            writer.write(np.zeros((*FRAME_SHAPE, 3), dtype=np.uint8))
        writer.release()

    return store_path


def run_main(
    trajectories_store: Path,
    monkeypatch: pytest.MonkeyPatch,
    predictor=None,
    extra_args: list[str] | None = None,
) -> Path:
    """Run `mask-tracked-crabs` with SAM2 replaced by the fake predictor.

    A None predictor makes loading one an error, which is how the tests
    that must fail *before* SAM2 is loaded assert that they do.
    """

    def fake_load(model_id: str, device: str):
        assert predictor is not None, "SAM2 should not have been loaded"
        return predictor

    monkeypatch.setattr("crabs.mask.mask_video.load_sam2_predictor", fake_load)
    output_dir = trajectories_store.parent / "mask_output"
    main(
        mask_parse_args(
            [
                "--boxes",
                str(trajectories_store),
                "--videos",
                str(trajectories_store.parent / "clips"),
                "--output_dir",
                str(output_dir),
                "--accelerator",
                "cpu",
                *(extra_args or []),
            ]
        )
    )
    return output_dir


def test_main(trajectories_store: Path, monkeypatch: pytest.MonkeyPatch):
    """The masks hold the crabs the trajectories store has, frame by frame.

    That comparison is the contract of this entry point: a frame-index
    error, a mis-assigned label and a dropped shard tail all fail it.
    """
    output_dir = run_main(trajectories_store, monkeypatch, FakePredictor())

    stores = list(output_dir.glob("*_masks_*.zarr"))
    assert len(stores) == 1
    assert stores[0].name.startswith("CrabTracks_masks_")

    masks = xr.open_datatree(stores[0], engine="zarr")["video"].to_dataset()
    tracks = xr.open_datatree(trajectories_store, engine="zarr")[
        "video"
    ].to_dataset()

    # the coordinates are the trajectories store's own, so the two stores
    # align 1:1 with no reindexing
    assert list(masks.clip_id.values) == list(tracks.clip_id.values)
    assert list(masks.individual.values) == list(tracks.individual.values)
    assert masks.labels.shape == (2, 3, *FRAME_SHAPE)
    assert masks.labels.dtype == np.uint16
    assert masks.attrs["boxes"]["source"] == "trajectories_zarr"

    # decode pixel values back to individuals through `label_of`, never by
    # position
    inverse = np.empty(int(masks.label_of.max()) + 1, dtype=int)
    inverse[masks.label_of.values] = np.arange(masks.sizes["individual"])

    for clip_index, clip_id in enumerate(masks.clip_id.values):
        n_clip_frames = len(read_tracked_bboxes_from_zarr(tracks, clip_id)[0])
        for time_index in range(n_clip_frames):
            frame = masks.labels.values[clip_index, time_index]
            in_masks = set(
                masks.individual.values[inverse[np.unique(frame[frame > 0])]]
            )
            in_tracks = set(
                tracks.position.isel(clip_id=clip_index, time=time_index)
                .dropna(dim="individual", how="all")
                .individual.values
            )
            assert in_masks == in_tracks, (clip_id, time_index)


@pytest.mark.parametrize(
    ("extra_args", "expected_error", "expected_message"),
    [
        # PR 2 widens the accepted suffixes to include .csv
        (["--boxes", "tracks.csv"], ValueError, "must be a trajectories"),
        (["--match", "nope*"], ValueError, "selected no video group"),
        (["--videos", "wrong_dir"], FileNotFoundError, "video-Loop00.mp4"),
    ],
)
def test_main_fails_before_sam2_is_loaded(
    trajectories_store: Path,
    monkeypatch: pytest.MonkeyPatch,
    extra_args: list[str],
    expected_error: type[Exception],
    expected_message: str,
):
    """A bad input costs a second, not a model download."""
    with pytest.raises(expected_error, match=expected_message):
        run_main(trajectories_store, monkeypatch, extra_args=extra_args)


def test_main_output_store_is_shared_across_runs(
    trajectories_store: Path, monkeypatch: pytest.MonkeyPatch
):
    """The array-job pattern: one run per video, all into one store.

    `--output_store` names the store exactly, so runs that would each
    pick their own timestamp write into the same one instead.
    """
    # add a second video group, and the clip videos it names
    tracks = xr.open_datatree(trajectories_store, engine="zarr")[
        "video"
    ].to_dataset()
    tracks.to_zarr(trajectories_store, group="video2")
    clips = trajectories_store.parent / "clips"
    for clip_video in sorted(clips.glob("video-*.mp4")):
        shutil.copy(
            clip_video, clips / clip_video.name.replace("video-", "video2-")
        )

    store_path = trajectories_store.parent / "CrabMasks-slurm42.zarr"
    for video_id in ["video", "video2"]:
        run_main(
            trajectories_store,
            monkeypatch,
            FakePredictor(),
            extra_args=[
                "--output_store",
                str(store_path),
                "--match",
                video_id,
                "--zarr_mode_store",
                "a",
            ],
        )

    masks = xr.open_datatree(store_path, engine="zarr")
    assert sorted(node.name for node in masks.leaves) == ["video", "video2"]
    for video_id in ["video", "video2"]:
        assert masks[video_id].labels.values.any()
