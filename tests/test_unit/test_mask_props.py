from pathlib import Path

import numpy as np
import pytest
import xarray as xr
import zarr

from crabs.mask.mask_props import (
    MASK_PROPS_VARS,
    SCALAR_VARS,
    _invert_label_of,
    add_mask_props_to_group,
    compute_clip_mask_props_dataset,
    frames_with_boxes,
)
from crabs.mask.mask_video import create_mask_store
from crabs.mask.utils.bbox_prompts import (
    BBOX_VARS,
    copy_video_bboxes_to_masks,
    extract_bboxes_from_clip,
)
from crabs.zarr.create_dataset import concat_clips
from scripts.generate_masks_from_bboxes import compute_mask_props_dataset

# Fixtures of the script's tests, reused for the agreement test
from tests.test_unit.test_generate_masks_from_bboxes import (  # noqa: F401
    EllipseLabels,
    ds_bboxes_annotations,
    ellipse_labels,
)

FRAME_SHAPE = (32, 32)  # height, width
INDIVIDUALS = ["id_0", "id_1", "id_2"]
ATTRS = {"occlusion_policy": "smallest_wins", "boxes": {"file": "t.zarr"}}
TIME_CHUNK = 2  # shorter than a clip


@pytest.fixture()
def ds_tracks(tmp_path: Path, clip_dataset_factory) -> xr.Dataset:
    """Return a two-clip video group of a trajectories store, read from zarr.

    Clip "Loop00" runs 4 frames: id_0 has a box in frames 0, 1 and 3, and
    id_2 in frame 0 only; frame 2 has no box. Clip "Loop01" runs 3 frames,
    with id_1 in every frame. `time` is chunked in chunks of 2 frames.
    """
    position_0 = np.full((4, 2, 3), np.nan)
    position_0[[0, 1, 3], :, 0] = 8.0
    position_0[0, :, 2] = 24.0
    position_1 = np.full((3, 2, 3), np.nan)
    position_1[:, :, 1] = 16.0

    ds_video = concat_clips(
        [
            clip_dataset_factory(
                "Loop00", position_0, np.full((4, 2, 3), 6.0)
            ),
            clip_dataset_factory(
                "Loop01", position_1, np.full((3, 2, 3), 6.0)
            ),
        ]
    )
    for var in BBOX_VARS:
        ds_video[var] = ds_video[var].astype(np.float32)
    ds_video.attrs["video_id"] = "video"
    store_path = tmp_path / "tracks.zarr"
    ds_video.chunk({"time": TIME_CHUNK}).to_zarr(store_path, group="video")
    return xr.open_zarr(store_path, group="video")


def paint_square(label_frame: np.ndarray, centre: int, half: int, label):
    """Paint a square of side `2 * half` centred at `centre` (row and col)."""
    label_frame[
        centre - half : centre + half, centre - half : centre + half
    ] = label


def paint_labels_at_boxes(
    ds_tracks: xr.Dataset, label_of: np.ndarray, n_frames: int
) -> np.ndarray:
    """Paint one square per box, of the individual's label.

    The half side of each square is 2 + its position along `individual`,
    so each individual has a distinct area.
    """
    n_clips = ds_tracks.sizes["clip_id"]
    labels = np.zeros((n_clips, n_frames, *FRAME_SHAPE), dtype=np.uint16)
    position = ds_tracks.position.values
    for c, t, k in zip(
        *np.nonzero(~np.isnan(position[:, :, 0, :])), strict=True
    ):
        centre = int(position[c, t, 0, k])
        paint_square(labels[c, t], centre, 2 + k, label_of[k])
    return labels


def create_masks_group_with_labels(
    store_path: Path,
    ds_tracks: xr.Dataset,
    n_frames: int | None = None,
    label_of: list[int] | None = None,
    copy_boxes: bool = True,
) -> np.ndarray:
    """Create a masks group, paint labels at the boxes and copy the boxes.

    Returns the painted label images.
    """
    n_frames = n_frames or ds_tracks.sizes["time"]
    labels_array = create_mask_store(
        store_path=store_path,
        video_id="video",
        clip_ids=[str(c) for c in ds_tracks.clip_id.values],
        n_frames=n_frames,
        individuals=INDIVIDUALS,
        image_shape=FRAME_SHAPE,
        metadata_dict=ATTRS,
        n_frames_per_shard=None,
        zarr_mode_group="w-",
    )
    if label_of is not None:
        zarr.open_group(store_path, use_consolidated=False)["video/label_of"][
            :
        ] = label_of
    label_of_values = np.asarray(label_of or [1, 2, 3])
    labels = paint_labels_at_boxes(ds_tracks, label_of_values, n_frames)
    labels_array[:] = labels
    if copy_boxes:
        copy_video_bboxes_to_masks(store_path, ds_tracks)
    return labels


def open_masks_group(store_path: Path) -> xr.Dataset:
    """Open the masks group as it is on disk."""
    return xr.open_zarr(store_path, group="video", consolidated=False)


def test_compute_clip_mask_props_dataset_agrees_with_script(
    ellipse_labels: EllipseLabels,  # noqa: F811
    ds_bboxes_annotations: xr.Dataset,  # noqa: F811
):
    """Check the copied measuring code agrees with that of PR #298.

    Both measure the same label images, with label = individual + 1.
    """
    label_images = ellipse_labels.label_images
    n_frames = len(label_images)

    # Measure with the script
    expected = compute_mask_props_dataset(label_images, ds_bboxes_annotations)

    # Measure with crabs
    ds_props = compute_clip_mask_props_dataset(
        label_images[np.newaxis],  # (1, n_frames, h, w)
        clip_index=0,
        time_slice=slice(0, n_frames),
        frames=np.arange(n_frames),
        individual_of_label=_invert_label_of(np.array([1, 2, 3])),
    )

    xr.testing.assert_allclose(
        ds_props,
        expected.drop_vars(list(expected.coords)).rename(
            {"image_id": "time", "id": "individual"}
        ),
    )
    assert all(ds_props[var].dtype == np.float32 for var in ds_props)


def test_compute_clip_mask_props_dataset_known_ellipse(
    ellipse_labels: EllipseLabels,  # noqa: F811
):
    """Check the parameters of an ellipse rotated by pi / 4."""
    # Measure frame 1, which holds a rotated ellipse of label 3
    ds_props = compute_clip_mask_props_dataset(
        ellipse_labels.label_images[np.newaxis],
        clip_index=0,
        time_slice=slice(1, 2),
        frames=np.array([1]),
        individual_of_label=_invert_label_of(np.array([1, 2, 3])),
    ).isel(time=0, individual=2)

    assert float(ds_props.ellipse_major_axis_theta) == pytest.approx(
        np.pi / 4, abs=0.02
    )
    assert float(ds_props.ellipse_major_axis_length) == pytest.approx(
        2 * ellipse_labels.semi_major_axis, rel=0.05
    )
    assert float(ds_props.ellipse_minor_axis_length) == pytest.approx(
        2 * ellipse_labels.semi_minor_axis, rel=0.05
    )
    np.testing.assert_allclose(
        ds_props.mask_centroid.values,
        [ellipse_labels.centre_x, ellipse_labels.centre_y],
        atol=0.01,
    )


def test_compute_clip_mask_props_dataset_unknown_label():
    """Check a label that is not in `label_of` raises an error."""
    labels = np.zeros((1, 1, *FRAME_SHAPE), dtype=np.uint16)
    paint_square(labels[0, 0], 10, 3, 9)  # 9 is not a crab's label

    with pytest.raises(ValueError, match=r"labels \[9\] that are not in"):
        compute_clip_mask_props_dataset(
            labels,
            clip_index=0,
            time_slice=slice(0, 1),
            frames=np.array([0]),
            individual_of_label=_invert_label_of(np.array([1, 2, 3])),
        )


def test_frames_with_boxes_matches_extract_bboxes_from_clip(
    ds_tracks: xr.Dataset,
):
    """Check the frames measured are those SAM2 is prompted on."""
    for clip_id in ds_tracks.clip_id.values:
        ds_clip = ds_tracks.sel(clip_id=clip_id)
        assert frames_with_boxes(ds_clip).tolist() == sorted(
            extract_bboxes_from_clip(ds_clip)
        )


def test_add_mask_props_to_group_measures_frames_with_boxes(
    ds_tracks: xr.Dataset, tmp_path: Path
):
    """Check only frames with a finite position are measured.

    Every frame of the label images is painted, including frames with no
    box and the padded frames past the end of the trajectories `time`, but
    those stay NaN.
    """
    # Create a masks group longer than the trajectories store
    store_path = tmp_path / "masks.zarr"
    n_frames = ds_tracks.sizes["time"] + 2
    create_masks_group_with_labels(store_path, ds_tracks, n_frames=n_frames)

    # Paint a mask of id_0 in every frame of every clip
    labels_array = zarr.open_group(store_path, use_consolidated=False)[
        "video/labels"
    ]
    labels = np.asarray(labels_array[:])
    labels[..., 0:4, 0:4] = 1
    labels_array[:] = labels

    # Measure
    add_mask_props_to_group(store_path, "video")

    # Check values are finite only in frames with boxes
    ds_masks = open_masks_group(store_path)
    frame_has_box = ds_masks.position.notnull().any(["space", "individual"])
    frame_has_mask = ds_masks.mask_area.notnull().any("individual")
    assert frame_has_box.any()
    assert not frame_has_box.all()
    xr.testing.assert_equal(frame_has_mask, frame_has_box)


def test_add_mask_props_to_group(ds_tracks: xr.Dataset, tmp_path: Path):
    """Check the 8 variables, their layout and their values.

    The values must equal those of one `compute_clip_mask_props_dataset`
    call over a whole clip, although the group is measured one time chunk
    at a time. The other arrays and the attrs must be unchanged.
    """
    # Create masks group
    store_path = tmp_path / "masks.zarr"
    create_masks_group_with_labels(store_path, ds_tracks)
    ds_before = open_masks_group(store_path).load()

    # Measure
    add_mask_props_to_group(store_path, "video")
    ds_masks = open_masks_group(store_path)

    # Check dims, dtype and chunks
    for var in SCALAR_VARS:
        assert ds_masks[var].dims == ("clip_id", "time", "individual")
        assert ds_masks[var].dtype == np.float32
        assert (
            ds_masks[var].encoding["chunks"]
            == ds_masks.confidence.encoding["chunks"]
        )
    assert ds_masks.mask_centroid.dims == (
        "clip_id",
        "time",
        "space",
        "individual",
    )
    assert ds_masks.mask_centroid.dtype == np.float32
    assert (
        ds_masks.mask_centroid.encoding["chunks"]
        == ds_masks.position.encoding["chunks"]
    )

    # Check values equal one call over each whole clip
    labels = ds_masks.labels.values
    n_frames = ds_masks.sizes["time"]
    for clip_index in range(ds_masks.sizes["clip_id"]):
        expected = compute_clip_mask_props_dataset(
            labels,
            clip_index,
            slice(0, n_frames),
            np.arange(n_frames),
            _invert_label_of(np.array([1, 2, 3])),
        )
        xr.testing.assert_equal(
            ds_masks[MASK_PROPS_VARS]
            .isel(clip_id=clip_index)
            .reset_coords(drop=True)
            .drop_vars(["time", "space", "individual"])
            .load(),
            expected,
        )

    # Check centroids sit at the box centres
    # (a square of even side 2h from c - h to c + h - 1 has mean c - 0.5)
    has_box = ds_masks.position.notnull()
    np.testing.assert_allclose(
        ds_masks.mask_centroid.where(has_box).values,
        (ds_masks.position - 0.5).values,
    )

    # Check other arrays and attrs are unchanged
    xr.testing.assert_identical(
        ds_masks[["labels", "label_of", *BBOX_VARS]].load(),
        ds_before[["labels", "label_of", *BBOX_VARS]],
    )
    assert ds_masks.attrs == ATTRS


def test_add_mask_props_to_group_label_of_not_k_plus_1(
    ds_tracks: xr.Dataset, tmp_path: Path
):
    """Check each value lands at its own individual, for any `label_of`."""
    # Create masks group with label_of = [3, 1, 2]
    store_path = tmp_path / "masks.zarr"
    create_masks_group_with_labels(store_path, ds_tracks, label_of=[3, 1, 2])

    # Measure
    add_mask_props_to_group(store_path, "video")

    # Check each individual's area is that of its own square, of half
    # side 2 + k
    ds_masks = open_masks_group(store_path)
    has_box = ds_masks.position.notnull().all("space")
    expected_area = (2 * (2 + np.arange(3))) ** 2
    expected = xr.DataArray(expected_area, dims="individual").broadcast_like(
        has_box
    )
    np.testing.assert_array_equal(
        ds_masks.mask_area.values,
        expected.where(has_box).values,
    )


@pytest.mark.parametrize(
    ("label_of", "copy_boxes", "expected_message"),
    [
        ([1, 1, 2], True, "unique, non-zero"),
        ([0, 1, 2], True, "unique, non-zero"),
        (None, False, "has no `position`"),
    ],
)
def test_add_mask_props_to_group_errors(
    ds_tracks: xr.Dataset,
    tmp_path: Path,
    label_of: list[int] | None,
    copy_boxes: bool,
    expected_message: str,
):
    """Check bad inputs raise an error before anything is written."""
    store_path = tmp_path / "masks.zarr"
    create_masks_group_with_labels(
        store_path, ds_tracks, label_of=label_of, copy_boxes=copy_boxes
    )

    with pytest.raises(ValueError, match=expected_message):
        add_mask_props_to_group(store_path, "video")

    ds_masks = open_masks_group(store_path)
    assert not set(MASK_PROPS_VARS) & set(ds_masks.data_vars)


def test_add_mask_props_to_group_overwrite(
    ds_tracks: xr.Dataset, tmp_path: Path
):
    """Check existing variables raise an error, unless `overwrite` is True."""
    # Create masks group and measure
    store_path = tmp_path / "masks.zarr"
    create_masks_group_with_labels(store_path, ds_tracks)
    add_mask_props_to_group(store_path, "video")

    # Erase all masks
    labels_array = zarr.open_group(store_path, use_consolidated=False)[
        "video/labels"
    ]
    labels_array[:] = 0

    # Check measuring again raises an error
    with pytest.raises(ValueError, match="already holds"):
        add_mask_props_to_group(store_path, "video")
    assert open_masks_group(store_path).mask_area.notnull().any()

    # Check overwriting replaces the values
    add_mask_props_to_group(store_path, "video", overwrite=True)
    ds_masks = open_masks_group(store_path)
    assert ds_masks.mask_area.isnull().all()
    assert ds_masks.attrs == ATTRS


def test_snippet_for_old_masks_stores(ds_tracks: xr.Dataset, tmp_path: Path):
    """Check the README snippet adds the boxes and the 8 variables.

    The old masks store is consolidated, as stores written before the
    boxes and the 8 variables were added. After the snippet, the
    consolidated metadata, which `xr.open_datatree` reads, shows them.
    """
    # Create an old, consolidated masks store
    masks_store = tmp_path / "masks.zarr"
    create_masks_group_with_labels(masks_store, ds_tracks, copy_boxes=False)
    zarr.consolidate_metadata(masks_store)
    trajectories_store = tmp_path / "tracks.zarr"

    # Run the snippet of crabs/mask/README.md
    dt_tracks = xr.open_datatree(trajectories_store, engine="zarr")
    for video_id in xr.open_datatree(masks_store, engine="zarr").children:
        copy_video_bboxes_to_masks(
            masks_store, dt_tracks[video_id].to_dataset()
        )
        add_mask_props_to_group(masks_store, video_id)
    zarr.consolidate_metadata(masks_store)

    # Check the group shows the new variables
    ds_masks = xr.open_datatree(masks_store, engine="zarr")[
        "video"
    ].to_dataset()
    assert {*BBOX_VARS, *MASK_PROPS_VARS} <= set(ds_masks.data_vars)
    assert ds_masks.mask_area.notnull().any()
    assert ds_masks.attrs == ATTRS
