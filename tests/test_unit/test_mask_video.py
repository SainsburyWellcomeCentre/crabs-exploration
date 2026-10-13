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
from crabs.mask.utils.boxes_from_dataset import extract_bboxes_from_clip
from crabs.zarr.create_dataset import concat_clips

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
    `predict_masks_on_device` calls, with the same return shapes. `_predict`
    returns (N, 1, H, W) boolean masks, for any N.

    We use it instead of a real SAM2 because its masks are known in advance,
    so tests can assert exact label pixels (e.g. overlaps). SAM2's masks
    are deterministic but can only be known by running the model.
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


def box_to_slices(box: np.ndarray) -> tuple[slice, slice]:
    """Convert an (x1, y1, x2, y2) box into image slices.

    Indexing an (H, W) image with the returned (row, column) slices
    selects the pixels inside the box, which are the pixels
    `FakePredictor` sets as the box's mask.
    """
    x1, y1, x2, y2 = np.asarray(box, dtype=int)
    return slice(y1, y2), slice(x1, x2)


def write_blank_video(path: Path, n_frames: int) -> None:
    """Write a video of `n_frames` black frames of `FRAME_SHAPE`."""
    writer = cv2.VideoWriter(
        str(path),
        cv2.VideoWriter_fourcc(*"mp4v"),
        10,
        FRAME_SHAPE[::-1],
    )
    for _ in range(n_frames):
        writer.write(np.zeros((*FRAME_SHAPE, 3), dtype=np.uint8))
    writer.release()


@pytest.fixture()
def blank_clip_video(tmp_path: Path) -> str:
    """Write a blank clip video."""
    path = tmp_path / "video-Loop00.mp4"
    write_blank_video(path, N_FRAMES)
    return str(path)


@pytest.fixture()
def masks_store_path_and_array(tmp_path: Path) -> tuple[Path, zarr.Array]:
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
        n_frames_per_shard=SHARD_N_FRAMES,
        zarr_mode_group="w-",
    )
    return store_path, labels_array


@pytest.fixture()
def trajectories_store_and_clips(tmp_path: Path, clip_dataset_factory) -> Path:
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
            clip_dataset_factory(
                "Loop00", position_0, np.full((3, 2, 2), 8.0)
            ),
            clip_dataset_factory(
                "Loop01", position_1, np.full((2, 2, 1), 8.0)
            ),
        ]
    )
    store_path = tmp_path / "CrabTracks.zarr"
    ds_video.to_zarr(store_path, group="video")

    videos_dir = tmp_path / "clips"
    videos_dir.mkdir()
    for clip_id, n_frames in [("Loop00", 3), ("Loop01", 2)]:
        write_blank_video(videos_dir / f"video-{clip_id}.mp4", n_frames)

    return store_path


@pytest.mark.parametrize("n_boxes", [1, 3])
def test_predict_masks_on_device_matches_sam2_predict(n_boxes: int):
    """Check SAM2's public `predict` matches the private-API path.

    We wrap the private-API path in predict_masks_on_device to keep
    the predicted masks on the GPU,

    The test runs a randomly initialised SAM2, so no checkpoint is downloaded:
    the masks are meaningless, but must be identical. A SAM2 bump that
    changes the private methods fails.
    """
    pytest.importorskip("sam2")
    from sam2.build_sam import build_sam2
    from sam2.sam2_image_predictor import SAM2ImagePredictor

    torch.manual_seed(0)
    rng = np.random.default_rng(0)

    # Initialise predictor
    predictor = SAM2ImagePredictor(
        build_sam2(
            "configs/sam2.1/sam2.1_hiera_t.yaml",
            ckpt_path=None,
            device="cpu",  # for CI
        )
    )

    # Pass an image
    predictor.set_image(rng.integers(0, 255, (48, 80, 3), dtype=np.uint8))

    # Define box prompts
    boxes = np.array(
        [
            [2.0, 3.0, 30.0, 40.0],
            [10.0, 5.0, 70.0, 45.0],
            [40.0, 20.0, 60.0, 30.0],
        ]
    )[:n_boxes]

    # Predict masks using public API
    masks_public, _, _ = predictor.predict(box=boxes, multimask_output=False)

    # Predict masks using our wrapper
    masks_on_device = predict_masks_on_device(predictor, boxes)

    # Check shape is as expected
    assert masks_on_device.shape == (n_boxes, 48, 80)

    # `predict` squeezes axis 0 and casts to float, so a single box comes back
    # as (1, H, W) and several as (N, 1, H, W): we reshape and cast here
    # to compare
    np.testing.assert_array_equal(
        masks_on_device.numpy(),
        masks_public.reshape(n_boxes, 48, 80).astype(bool),
    )


def test_create_mask_store(tmp_path: Path):
    """Check the initialised mask store, with no data written yet."""
    # Initialise mask store
    store_path = tmp_path / "masks.zarr"
    labels_array = create_mask_store(
        store_path=store_path,
        video_id="video",
        clip_ids=["Loop00", "Loop01"],
        n_frames=N_FRAMES,
        individuals=INDIVIDUALS,
        image_shape=FRAME_SHAPE,
        metadata_dict=ATTRS,
        n_frames_per_shard=SHARD_N_FRAMES,
        zarr_mode_group="w-",
    )
    height, width = FRAME_SHAPE

    # Check it is as expected
    assert labels_array.shape == (2, N_FRAMES, height, width)
    assert labels_array.dtype == np.uint16
    assert labels_array.chunks == (1, 1, height, width)
    assert labels_array.shards == (1, SHARD_N_FRAMES, height, width)

    # Check zarr's default compressor
    assert [type(c) for c in labels_array.compressors] == [ZstdCodec]

    # Check metadata exists in store, but not data
    # (each clip fills its own region afterwards)
    assert (store_path / "video/labels/zarr.json").is_file()
    assert not (store_path / "video/labels/c").exists()

    # Open as dataset and check coordinates are as expected
    ds = xr.open_datatree(store_path, engine="zarr")["video"].to_dataset()
    assert ds.labels.dims == ("clip_id", "time", "img_h", "img_w")
    assert list(ds.clip_id.values) == ["Loop00", "Loop01"]
    assert list(ds.individual.values) == INDIVIDUALS
    assert ds.label_of.dims == ("individual",)
    # one label per individual, from 1 since 0 is the background
    assert list(ds.label_of.values) == [1, 2, 3]

    # Check dataset attrs are as expected
    assert ds.attrs == ATTRS


@pytest.mark.parametrize("max_prompts_per_batch", [1, 2, 8])
def test_write_clip_masks_to_store(
    masks_store_path_and_array: tuple[Path, zarr.Array],
    blank_clip_video: str,
    max_prompts_per_batch: int,
):
    """Test mask prediction and writing for one video clip.

    The written frames must not depend on `max_prompts_per_batch`: it is
    a memory knob, so it must not change the pixels.
    """
    # Get data from initialised masks store
    store_path, labels_array = masks_store_path_and_array
    label_of = xr.open_datatree(store_path, engine="zarr")["video"].label_of

    # Define prompt boxes
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

    # Instantiate fake predictor (predicted mask = box)
    predictor = FakePredictor()

    # Run prediction on clip 0
    n_frames_read = write_clip_masks_to_store(
        video_clip_path=blank_clip_video,
        tracked_bboxes_dict=tracked_bboxes_dict,
        labels_array=labels_array,
        clip_index=0,
        label_of=label_of,
        predictor=predictor,
        max_prompts_per_batch=max_prompts_per_batch,
        n_frames_per_shard=SHARD_N_FRAMES,
    )

    # check frames read
    assert n_frames_read == N_FRAMES
    # SAM2 is skipped entirely on a frame with no boxes
    assert predictor.n_frames_seen == 3

    # Inspect labels for clip 0
    labels_clip_0 = np.asarray(labels_array[0])

    # frame 0: smallest_wins, so id_2 owns the contested pixels
    boxes_frame_0 = tracked_bboxes_dict[0]["tracked_boxes"]
    expected = np.zeros(FRAME_SHAPE, dtype=np.uint16)
    expected[box_to_slices(boxes_frame_0[1])] = 1  # id_0
    expected[box_to_slices(boxes_frame_0[0])] = 3  # id_2, painted last
    np.testing.assert_array_equal(labels_clip_0[0], expected)

    # frames with no boxes are all background (label 0)
    assert not labels_clip_0[1].any()
    assert not labels_clip_0[2].any()

    # single-box frames:
    # - frame 3 is the only frame with id_1, and it shares a shard with
    #   empty frame 2
    # - frame 4 is the part-filled shard tail
    for frame_index, label in [(3, 2), (4, 1)]:  # id_1, id_0
        box = tracked_bboxes_dict[frame_index]["tracked_boxes"][0]
        expected = np.zeros(FRAME_SHAPE, dtype=np.uint16)
        expected[box_to_slices(box)] = label
        np.testing.assert_array_equal(
            labels_clip_0[frame_index],
            expected,
            err_msg=f"frame {frame_index}",
        )

    # check the other clip is untouched by this one's writes
    # and all of its labels are 0
    assert not np.asarray(labels_array[1]).any()


def run_main_with_fake_predictor(
    trajectories_store: Path,
    monkeypatch: pytest.MonkeyPatch,
    predictor=None,
    extra_args: list[str] | None = None,
) -> Path:
    """Run `mask-tracked-crabs` using the fake predictor rather than SAM2.

    A None predictor makes loading one an error, which is how the tests
    that must fail *before* SAM2 is loaded assert that they do.
    """

    # Monkeypatch load_sam2_predictor
    def fake_load(model_id: str, device: str):
        assert predictor is not None, "SAM2 should not have been loaded"
        return predictor

    monkeypatch.setattr("crabs.mask.mask_video.load_sam2_predictor", fake_load)

    # Run main and return output dir
    output_dir = trajectories_store.parent / "mask_output"
    main(
        mask_parse_args(
            [
                "--boxes",
                str(trajectories_store),
                "--videos_dir",
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


def test_main(
    trajectories_store_and_clips: Path,
    monkeypatch: pytest.MonkeyPatch,
):
    """The masks hold the crabs the trajectories store has, frame by frame.

    That comparison is the contract of this entry point: a frame-index
    error, a mis-assigned label and a dropped shard tail all fail it.
    """
    output_dir = run_main_with_fake_predictor(
        trajectories_store_and_clips,
        monkeypatch,
        FakePredictor(),
    )

    stores = list(output_dir.glob("*_masks_*.zarr"))
    assert len(stores) == 1
    assert stores[0].name.startswith("CrabTracks_masks_")

    masks = xr.open_datatree(stores[0], engine="zarr")["video"].to_dataset()
    tracks = xr.open_datatree(trajectories_store_and_clips, engine="zarr")[
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
        n_clip_frames = len(
            extract_bboxes_from_clip(tracks.sel(clip_id=clip_id))
        )
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
        (["--videos_dir", "wrong_dir"], FileNotFoundError, "video-Loop00.mp4"),
    ],
)
def test_main_fails_before_sam2_is_loaded(
    trajectories_store_and_clips: Path,
    monkeypatch: pytest.MonkeyPatch,
    extra_args: list[str],
    expected_error: type[Exception],
    expected_message: str,
):
    """A bad input costs a second, not a model download."""
    with pytest.raises(expected_error, match=expected_message):
        run_main_with_fake_predictor(
            trajectories_store_and_clips,
            monkeypatch,
            extra_args=extra_args,
        )


def test_main_output_store_is_shared_across_runs(
    trajectories_store_and_clips: Path, monkeypatch: pytest.MonkeyPatch
):
    """The array-job pattern: one run per video, all into one store.

    `--output_store` names the store exactly, so runs that would each
    pick their own timestamp write into the same one instead.
    """
    # add a second video group, and the clip videos it names
    tracks = xr.open_datatree(trajectories_store_and_clips, engine="zarr")[
        "video"
    ].to_dataset()
    tracks.to_zarr(trajectories_store_and_clips, group="video2")
    clips = trajectories_store_and_clips.parent / "clips"
    for clip_video in sorted(clips.glob("video-*.mp4")):
        shutil.copy(
            clip_video, clips / clip_video.name.replace("video-", "video2-")
        )

    store_path = trajectories_store_and_clips.parent / "CrabMasks-slurm42.zarr"
    for video_id in ["video", "video2"]:
        run_main_with_fake_predictor(
            trajectories_store_and_clips,
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
