from pathlib import Path

import numpy as np
import pytest
import xarray as xr

from scripts.generate_masks_from_bboxes import create_dataset_store


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
            "space": np.array(["x", "y"], dtype=object),  # as main casts it
            "id": range(3),
        },
        attrs={
            "images_directories": Path("/data/frames"),
            "map_image_id_to_filename": {5: "a.png", 7: "b.png"},
        },
    )


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
