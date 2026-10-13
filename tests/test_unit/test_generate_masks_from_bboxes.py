from pathlib import Path

import numpy as np
import pytest
import xarray as xr
from skimage.draw import ellipse

from scripts.generate_masks_from_bboxes import (
    compute_mask_props,
    create_dataset_store,
)


@pytest.fixture
def ds_bboxes():
    """Two images, three id slots, with attributes as ethology sets them."""
    return xr.Dataset(
        {
            "position": (
                ("image_id", "space", "id"),
                np.full((2, 2, 3), np.nan),
            )
        },
        coords={
            "image_id": [5, 7],
            "space": ["x", "y"],
            "id": range(3),
        },
        attrs={
            "images_directories": Path("/data/frames"),
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
        ds.ellipse_major_axis_theta.sel(image_id=7, id=2),
        np.pi / 4,
        atol=0.02,
    )
    # every other slot has no mask
    for name in ds.data_vars:
        assert int(ds[name].isnull().sum()) == 4 * ds[name].sizes.get(
            "space", 1
        )


def test_store_round_trip(tmp_path, ds_bboxes, labels):
    """Run the three passes of main, without SAM2."""
    store = tmp_path / "bboxes_and_masks.zarr"
    labels_zarr = create_dataset_store(store, ds_bboxes, (100, 120), {})
    labels_zarr[:] = labels  # what predict_masks_across_images does
    compute_mask_props(labels_zarr, ds_bboxes).to_zarr(store, mode="a")

    ds = xr.open_zarr(store)
    mask = ds.labels.sel(image_id=7) == ds.label_of.sel(id=2)
    assert int(mask.sum()) == ds.mask_area.sel(image_id=7, id=2)
    np.testing.assert_array_equal(ds.labels.values, labels)


def test_create_dataset_store(tmp_path, ds_bboxes):
    """Check the template has zero labels, label_of and zarr-safe attrs."""
    store = tmp_path / "bboxes_and_masks.zarr"
    labels_zarr = create_dataset_store(
        store,
        ds_bboxes,
        (100, 120),
        {"occlusion_policy": "smallest_wins"},
    )

    # returned array is the labels array, one frame per chunk
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
