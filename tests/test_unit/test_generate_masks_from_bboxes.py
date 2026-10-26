from pathlib import Path

import numpy as np
import pytest
import xarray as xr
import zarr
from PIL import Image
from skimage.draw import ellipse

from scripts.generate_masks_from_bboxes import (
    ImageArrayLazy,
    append_mask_props,
    compute_mask_props,
    create_dataset_store,
    flatten_masks_smallest_wins,
    get_bboxes_and_labels_per_frame,
)


@pytest.fixture
def ds_bboxes():
    """Two images, three id slots, as ethology >= 0.1.4 returns them.

    ethology's ``image_id`` is the image's index in the sorted list of all
    images in the annotation file, so images without annotations leave gaps.
    "image_id": [5, 7] mimics that, and make a mix-up between image ID and
    position along ``image_id``.
    """
    return xr.Dataset(
        {
            "position": (
                ("image_id", "space", "id"),
                np.full((2, 2, 3), np.nan),
            ),
            "shape": (
                ("image_id", "space", "id"),
                np.full((2, 2, 3), np.nan),
            ),
            "image_shape": (
                ("image_id", "space"),
                np.full((2, 2), [120, 100]),
            ),
            "category": (("image_id", "id"), np.full((2, 3), -1)),
        },
        coords={
            "image_id": [5, 7],
            "space": np.array(["x", "y"], dtype=object),
            "id": range(3),
        },
        attrs={
            "annotation_files": Path("/data/annotations/coco.json"),
            "annotation_format": "COCO",
            "images_directories": Path("/data/frames"),
            "map_category_to_str": {1: "crab"},
            "map_image_id_to_filename": {5: "a.png", 7: "b.png"},
        },
    )


@pytest.fixture
def labels():
    """(2, 100, 120) label images: id 0 in image 5, id 2 in image 7."""
    lab = np.zeros((2, 100, 120), np.uint16)
    rr, cc = ellipse(50, 60, 8, 20)  # semi-axes 8 (rows), 20 (cols)
    lab[0, rr, cc] = 1  # elongated along x
    # skimage.draw rotates from +x towards -y, so this is theta = pi / 4
    rr, cc = ellipse(50, 60, 8, 20, rotation=-np.pi / 4)
    lab[1, rr, cc] = 3
    return lab


def test_compute_mask_props(labels, ds_bboxes):
    """Check values land at the right (image_id, id) and are correct."""
    ds = compute_mask_props(labels, ds_bboxes)

    first = ds.sel(image_id=5, id=0)
    assert first.mask_area == (labels[0] == 1).sum()
    np.testing.assert_allclose(first.mask_centroid, [60, 50])  # (x, y)
    # axis lengths are full lengths, not semi-axes
    np.testing.assert_allclose(first.ellipse_major_axis_length, 40, rtol=0.05)
    np.testing.assert_allclose(first.ellipse_minor_axis_length, 16, rtol=0.1)
    np.testing.assert_allclose(first.ellipse_major_axis_theta, 0, atol=0.02)
    np.testing.assert_allclose(
        first.ellipse_eccentricity,
        np.sqrt(
            1
            - (
                first.ellipse_minor_axis_length
                / first.ellipse_major_axis_length
            )
            ** 2
        ),
        atol=1e-3,
    )
    np.testing.assert_allclose(
        ds.ellipse_major_axis_theta.sel(image_id=7, id=2),
        np.pi / 4,
        atol=0.02,
    )
    # every other slot has no mask
    for name in ds.data_vars:
        assert int(ds[name].isnull().sum()) == 4 * ds[name].sizes.get(
            "space", 1
        )


@pytest.mark.parametrize(
    "space_dtype",
    [
        object,  # as from ethology >= 0.1.4
        "<U1",  # as built e.g. coords={"space": ["x", "y"]}
    ],
)
def test_store_round_trip(tmp_path, ds_bboxes, labels, space_dtype):
    """Run the three passes of main, without SAM2."""
    ds_bboxes = ds_bboxes.assign_coords(
        space=ds_bboxes.space.astype(space_dtype)
    )
    store = tmp_path / "bboxes_and_masks.zarr"
    create_dataset_store(
        store, ds_bboxes, (100, 120), {"occlusion_policy": "smallest_wins"}
    )
    labels_zarr = zarr.open_group(store)["labels"]
    labels_zarr[:] = labels  # what predict_masks_across_images does
    ds_props = compute_mask_props(labels_zarr, ds_bboxes)
    append_mask_props(store, ds_props)

    ds = xr.open_zarr(store)
    mask = ds.labels.sel(image_id=7) == ds.label_of.sel(id=2)
    assert int(mask.sum()) == ds.mask_area.sel(image_id=7, id=2)
    np.testing.assert_array_equal(ds.labels.values, labels)
    # root attributes survive the append
    assert ds.attrs["map_image_id_to_filename"] == {"5": "a.png", "7": "b.png"}
    assert ds.attrs["occlusion_policy"] == "smallest_wins"


def test_create_dataset_store(tmp_path, ds_bboxes):
    """Check the template has zero labels, label_of and zarr-safe attrs."""
    store = tmp_path / "bboxes_and_masks.zarr"
    create_dataset_store(
        store,
        ds_bboxes,
        (100, 120),
        {"occlusion_policy": "smallest_wins"},
    )

    # labels array has one frame per chunk
    labels_zarr = zarr.open_group(store)["labels"]
    assert labels_zarr.shape == (2, 100, 120)
    assert labels_zarr.chunks == (1, 100, 120)

    ds = xr.open_zarr(store)
    # no labels chunk is written, so they read back as the fill value 0
    assert int(ds.labels.max()) == 0
    assert ds.labels.dtype == np.uint16
    assert list(ds.label_of.values) == [1, 2, 3]
    # bbox variables and coordinates are written
    assert list(ds.image_id.values) == [5, 7]
    assert "position" in ds
    # attributes are zarr-safe; int dict keys come back as strings
    assert ds.attrs["images_directories"] == "/data/frames"
    assert ds.attrs["map_image_id_to_filename"] == {"5": "a.png", "7": "b.png"}
    assert ds.attrs["occlusion_policy"] == "smallest_wins"


def test_create_dataset_store_too_many_ids(tmp_path):
    """Check more id slots than uint16 labels can hold raise an error."""
    n_ids = np.iinfo(np.uint16).max + 1
    ds_bboxes = xr.Dataset(
        {"position": (("image_id", "id"), np.full((1, n_ids), np.nan))},
        coords={"image_id": [0], "space": ["x", "y"], "id": range(n_ids)},
    )
    with pytest.raises(ValueError, match="do not fit in uint16"):
        create_dataset_store(tmp_path / "s.zarr", ds_bboxes, (10, 10), {})


def test_image_array_lazy_different_sizes(tmp_path):
    """Check images of different sizes raise before any is loaded."""
    paths = [tmp_path / "a.png", tmp_path / "b.png"]
    Image.new("RGB", (120, 100)).save(paths[0])
    Image.new("RGB", (60, 100)).save(paths[1])
    with pytest.raises(ValueError, match="different sizes"):
        ImageArrayLazy(iter(paths))


def test_flatten_masks_smallest_wins():
    """Check overlapping pixels take the label of the smaller mask."""
    masks = np.zeros((3, 10, 10), bool)
    masks[0, 2:8, 2:8] = True  # small, 36 pixels
    masks[1, :, :6] = True  # large, 60 pixels, overlaps mask 0
    masks[2, 8:, 6:] = True  # 8 pixels, no overlap
    label_frame = flatten_masks_smallest_wins(masks, [4, 7, 9])

    assert label_frame.dtype == np.uint16
    assert (label_frame[masks[0]] == 4).all()
    assert (label_frame[masks[1] & ~masks[0]] == 7).all()
    assert (label_frame[masks[2]] == 9).all()
    assert (label_frame[~masks.any(axis=0)] == 0).all()


def test_flatten_masks_equal_areas():
    """Check equal-area masks are painted in input order: last one wins."""
    masks = np.zeros((2, 10, 10), bool)
    masks[0, :, :6] = True
    masks[1, :, 4:] = True
    label_frame = flatten_masks_smallest_wins(masks, [1, 2])
    assert (label_frame[:, 4:6] == 2).all()


@pytest.mark.parametrize(
    "dims",
    [("image_id", "space", "id"), ("image_id", "id", "space")],
)
def test_get_bboxes_mask_labels_per_frame(dims):
    """Check boxes and labels per frame, whatever the dimension order."""
    position = np.full((2, 2, 3), np.nan)  # image_id, space, id
    shape = np.full((2, 2, 3), np.nan)
    position[0, :, 0], shape[0, :, 0] = [10, 20], [4, 6]
    position[0, :, 2], shape[0, :, 2] = [50, 60], [10, 8]
    ds_bboxes = xr.Dataset(
        {
            "position": (("image_id", "space", "id"), position),
            "shape": (("image_id", "space", "id"), shape),
        },
        coords={"image_id": [5, 7], "space": ["x", "y"], "id": range(3)},
    ).transpose(*dims)

    boxes_and_labels = get_bboxes_and_labels_per_frame(ds_bboxes)

    # frame 0: slots 0 and 2 filled, as [x1, y1, x2, y2]
    boxes, labels = boxes_and_labels[0]
    np.testing.assert_array_equal(boxes, [[8, 17, 12, 23], [45, 56, 55, 64]])
    np.testing.assert_array_equal(labels, [1, 3])
    # frame 1: no boxes
    boxes, labels = boxes_and_labels[1]
    assert boxes.shape == (0, 4)
    assert labels.shape == (0,)
