from pathlib import Path
from typing import NamedTuple

import numpy as np
import pytest
import xarray as xr
import zarr
from PIL import Image
from skimage.draw import ellipse

from scripts.orientation.generate_masks_from_bboxes import (
    ImageArrayLazy,
    append_mask_props,
    compute_mask_props_dataset,
    create_dataset_store,
    flatten_masks_smallest_wins,
    get_bboxes_and_label_ids_per_frame,
)


@pytest.fixture
def ds_bboxes_annotations() -> xr.Dataset:
    """Get a sample `ethology` annotations bounding box dataset.

    Two images, three id slots, as ethology >= 0.1.4 returns them.

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


class EllipseLabels(NamedTuple):
    """Label image array and the ellipses drawn in them, in pixels."""

    label_images: np.ndarray
    centre_x: int
    centre_y: int
    semi_major_axis: int
    semi_minor_axis: int


@pytest.fixture
def ellipse_labels(ds_bboxes_annotations) -> EllipseLabels:
    """Get label image arrays with one elliptical mask per image.

    Mimics the ``labels`` array that ``predict_masks_across_images``
    writes to the zarr store for the two images in the
    ``ds_bboxes_annotations`` fixture. Each pixel holds the label
    of the mask it belongs to (label = id + 1), with 0 as background.

    The ellipses have known centroid, axis lengths and orientation, so
    we can check the output of ``compute_mask_props_dataset``:

    - image_id 5 (index 0): id 0, major axis along x (theta = 0).
    - image_id 7 (index 1): id 2, same ellipse rotated to theta = pi / 4.
    """
    # Get shape of array from bboxes dataset
    n_images = ds_bboxes_annotations.sizes["image_id"]
    image_shape = ds_bboxes_annotations.image_shape.isel(image_id=0)
    width = int(image_shape.sel(space="x"))
    height = int(image_shape.sel(space="y"))

    # Define an ellipse centred in the image, with major axis along x
    centre_x, centre_y = width // 2, height // 2
    semi_major_axis, semi_minor_axis = 20, 8  # pixels

    # Initialise label images array
    label_images = np.zeros((n_images, height, width), np.uint16)

    # Define masks
    # image_id 5: id 0 (label 1), unrotated
    rows, cols = ellipse(
        centre_y,  # row
        centre_x,  # col
        r_radius=semi_minor_axis,
        c_radius=semi_major_axis,
    )
    label_images[0, rows, cols] = 1

    # image_id 7: id 2 (label 3), rotated.
    # skimage.draw rotates from +x towards -y, so this is theta = pi / 4
    rows, cols = ellipse(
        centre_y,  # row
        centre_x,  # col
        r_radius=semi_minor_axis,
        c_radius=semi_major_axis,
        rotation=-np.pi / 4,
    )
    label_images[1, rows, cols] = 3

    return EllipseLabels(
        label_images,
        centre_x,
        centre_y,
        semi_major_axis,
        semi_minor_axis,
    )


def test_create_dataset_store(tmp_path, ds_bboxes_annotations):
    """Check the initialised store is as expected.

    It should have zero labels, label_of and zarr-safe attrs.
    """
    # Initialise store
    n_images = ds_bboxes_annotations.sizes["image_id"]
    image_h, image_w = 100, 120
    metadata = {"occlusion_policy": "smallest_wins"}

    zarr_store = tmp_path / "bboxes_and_masks.zarr"
    create_dataset_store(
        zarr_store,
        ds_bboxes_annotations,
        image_shape=(image_h, image_w),
        metadata_dict=metadata,
    )

    # Inspect dataset from store
    ds_init = xr.open_zarr(zarr_store)

    # check labels array
    assert ds_init.labels.shape == (n_images, image_h, image_w)
    # one frame per chunk on disk
    assert ds_init.labels.encoding["chunks"] == (1, image_h, image_w)

    # check no labels chunk is written, so they read back as the fill value 0
    assert int(ds_init.labels.max()) == 0
    assert ds_init.labels.dtype == np.uint16

    # check label_of maps each id slot to its label (id + 1; 0 is background)
    assert list(ds_init.label_of.values) == [1, 2, 3]

    # check the bbox variables and coordinates are written unchanged,
    # with image_id values kept (not reset to positions)
    xr.testing.assert_equal(
        ds_init[list(ds_bboxes_annotations.data_vars)],
        ds_bboxes_annotations.drop_attrs(),
    )

    # check attributes are zarr-safe; int dict keys come back as strings
    assert ds_init.attrs["images_directories"] == "/data/frames"
    assert ds_init.attrs["map_image_id_to_filename"] == {
        "5": "a.png",
        "7": "b.png",
    }
    assert ds_init.attrs["occlusion_policy"] == metadata["occlusion_policy"]


def test_create_dataset_store_too_many_ids(tmp_path):
    """Check an error is raised if the labels do not fit in uint16.

    Labels are id + 1 (0 is background), so the largest uint16 value
    (65535) fits 65535 id slots. With one more, the last label would
    wrap around to 0 and its mask would be lost as background.
    """
    # Define a bboxes dataset with one id slot too many
    n_ids = np.iinfo(np.uint16).max + 1
    ds_bboxes = xr.Dataset(
        {
            "position": (
                ("image_id", "id"),
                np.full(
                    (1, n_ids),
                    np.nan,
                ),
            ),
        },
        coords={
            "image_id": [0],
            "space": ["x", "y"],
            "id": range(n_ids),
        },
    )

    # Check that initialising the output zarr store with it
    # raises an error
    with pytest.raises(ValueError, match="do not fit in uint16"):
        create_dataset_store(
            tmp_path / "bboxes_and_masks.zarr",
            ds_bboxes,
            image_shape=(10, 10),
            metadata_dict={},
        )


@pytest.mark.parametrize(
    "dims",
    [
        ("image_id", "space", "id"),
        ("image_id", "id", "space"),
    ],
)
def test_get_bboxes_and_label_ids_per_frame(dims):
    """Test output of get_bboxes_and_label_ids_per_frame.

    The output is a dict mapping position allong image_id to
    boxes and label IDs.

    Boxes go from centre and size to [x1, y1, x2, y2] corners. Each
    box's label ID is its id + 1. Frames are keyed by their position along
    ``image_id`` (0, 1), not by their image_id value (5, 7).

    Parametrised over the dimension order, which must not change the
    result.
    """
    # Define two boxes in the first image, at id slots 0 and 2,
    # as {id: (centre (x, y), size (width, height))}
    # (id slot 1 and the second image have no boxes, so are NaN)
    boxes_first_image = {
        0: ([10, 20], [4, 6]),
        2: ([50, 60], [10, 8]),
    }

    # Build dataset with these boxes
    position = np.full((2, 2, 3), np.nan)  # (image_id, space, id)
    shape = np.full((2, 2, 3), np.nan)
    for id_slot, (centre, size) in boxes_first_image.items():
        position[0, :, id_slot] = centre
        shape[0, :, id_slot] = size

    ds_bboxes = xr.Dataset(
        {
            "position": (("image_id", "space", "id"), position),
            "shape": (("image_id", "space", "id"), shape),
        },
        coords={
            "image_id": [5, 7],
            "space": ["x", "y"],
            "id": range(3),
        },
    ).transpose(*dims)

    # Compute boxes and label IDs per frame
    boxes_and_label_ids = get_bboxes_and_label_ids_per_frame(ds_bboxes)

    # Check keys are the position along image_id
    assert list(boxes_and_label_ids.keys()) == [0, 1]

    # Check boxes in first frame
    # corners as:
    # [x1, y1, x2, y2] = [x - w/2, y - h/2, x + w/2, y + h/2],
    boxes, label_ids = boxes_and_label_ids[0]
    np.testing.assert_array_equal(
        boxes,
        [
            [8, 17, 12, 23],  # id 0
            [45, 56, 55, 64],  # id 2
        ],
    )
    # with label IDs id + 1
    np.testing.assert_array_equal(
        label_ids,
        [id_slot + 1 for id_slot in boxes_first_image],
    )

    # check second frame has no boxes
    boxes, label_ids = boxes_and_label_ids[1]
    assert boxes.shape == (0, 4)
    assert label_ids.shape == (0,)


def test_compute_mask_props(ellipse_labels, ds_bboxes_annotations):
    """Check computation of mask properties are correct and aligned.

    They should land at the right (image_id, id) and be correct.
    """
    # Compute mask properties dataset
    ds_props = compute_mask_props_dataset(
        ellipse_labels.label_images,
        ds_bboxes_annotations,
    )

    # Select a single mask (first ellipse)
    ds_single_mask = ds_props.sel(image_id=5, id=0)

    # check properties of first ellipse
    # mask area
    assert (
        ds_single_mask.mask_area == (ellipse_labels.label_images[0] == 1).sum()
    )
    # check centroid, as (x, y) along "space"
    np.testing.assert_allclose(
        ds_single_mask.mask_centroid,
        [ellipse_labels.centre_x, ellipse_labels.centre_y],
    )
    # check axes lengths
    np.testing.assert_allclose(
        ds_single_mask.ellipse_major_axis_length,
        2 * ellipse_labels.semi_major_axis,
        rtol=0.05,
    )
    np.testing.assert_allclose(
        ds_single_mask.ellipse_minor_axis_length,
        2 * ellipse_labels.semi_minor_axis,
        rtol=0.1,
    )
    # check orientation (theta)
    np.testing.assert_allclose(
        ds_single_mask.ellipse_major_axis_theta,
        0,
        atol=0.02,
    )
    # check eccentricity
    np.testing.assert_allclose(
        ds_single_mask.ellipse_eccentricity,
        np.sqrt(
            1
            - (
                ds_single_mask.ellipse_minor_axis_length
                / ds_single_mask.ellipse_major_axis_length
            )
            ** 2
        ),
        atol=1e-3,
    )

    # check theta in second ellipse
    # Note: second ellipse is same as first but rotated
    np.testing.assert_allclose(
        ds_props.ellipse_major_axis_theta.sel(image_id=7, id=2),
        np.pi / 4,
        atol=0.02,
    )

    # check only the slots with a mask have values
    has_mask = xr.zeros_like(ds_props.mask_area, dtype=bool)
    has_mask.loc[{"image_id": 5, "id": 0}] = True
    has_mask.loc[{"image_id": 7, "id": 2}] = True
    for da in ds_props.data_vars.values():
        if "space" in da.dims:  # e.g. mask_centroid
            has_value = da.notnull().all("space")
        else:
            has_value = da.notnull()
        xr.testing.assert_equal(has_value, has_mask)


@pytest.mark.parametrize(
    "space_dtype",
    [
        object,  # as from ethology >= 0.1.4
        "<U1",  # as built e.g. coords={"space": ["x", "y"]}
    ],
)
def test_append_mask_props(
    tmp_path,
    ds_bboxes_annotations,
    ellipse_labels,
    space_dtype,
):
    """Check appending to zarr store without SAM2.

    Creates the store, writes the label images in place of the SAM2
    predictions, then computes and appends the mask properties. Checks
    the labels, mask properties and root attributes read back correctly.

    Parametrised over the input ``space`` dtype: both input dtypes
    must work.
    """
    # Assign relevant type to space coord
    ds_bboxes_annotations = ds_bboxes_annotations.assign_coords(
        space=ds_bboxes_annotations.space.astype(space_dtype)
    )

    # Create zarr store with dataset
    zarr_path = tmp_path / "bboxes_and_masks.zarr"
    metadata = {"occlusion_policy": "smallest_wins"}
    create_dataset_store(
        zarr_path,
        ds_bboxes_annotations,
        ellipse_labels.label_images.shape[1:],
        metadata_dict=metadata,
    )

    # Write label images from the fixture
    # in place of SAM2's predictions
    # (as predict_masks_across_images does)
    labels_zarr = zarr.open_group(zarr_path)["labels"]
    labels_zarr[:] = ellipse_labels.label_images

    # Compute mask properties and append to the store
    ds_props = compute_mask_props_dataset(labels_zarr, ds_bboxes_annotations)
    append_mask_props(zarr_path, ds_props)

    # Inspect zarr store after appending mask properties;
    # (compare to the fixture data)
    ds = xr.open_zarr(zarr_path)

    # check attributes survive the append
    assert ds.attrs["map_image_id_to_filename"] == {"5": "a.png", "7": "b.png"}
    assert ds.attrs["occlusion_policy"] == metadata["occlusion_policy"]

    # compare labels in dataset against the fixture
    np.testing.assert_array_equal(
        ds.labels.values,
        ellipse_labels.label_images,
    )

    # check the store is self-consistent: for every (image_id, id),
    # the pixel count of its mask, found via label_of, matches the
    # stored mask_area (NaN for slots without a mask)
    pixel_counts = (ds.labels == ds.label_of).sum(["img_h", "img_w"])
    xr.testing.assert_equal(
        pixel_counts.where(pixel_counts > 0),  # > 0 turns 0s into NaNs
        ds.mask_area.transpose(*pixel_counts.dims),
    )


def test_image_array_lazy_different_sizes(tmp_path):
    """Check images of different sizes raise an error."""
    image_paths = [
        tmp_path / "a.png",
        tmp_path / "b.png",
    ]
    # Create mock images at those paths with different sizes
    Image.new("RGB", (120, 100)).save(image_paths[0])
    Image.new("RGB", (60, 100)).save(image_paths[1])

    # Check that defining an image lazy array raises
    with pytest.raises(ValueError, match="different sizes"):
        ImageArrayLazy(iter(image_paths))


def test_flatten_masks_smallest_wins():
    """Check overlapping pixels take the label of the smaller mask."""
    # Define an array with overlapping masks
    masks = np.zeros((3, 10, 10), bool)
    masks[0, 2:8, 2:8] = True  # small, 36 pixels, wins in overlap
    masks[1, :, :6] = True  # large, 60 pixels, overlaps mask 0
    masks[2, 8:, 6:] = True  # 8 pixels, no overlap

    # Define arbitrary label IDs (not 1, 2, 3), to check masks are
    # painted with the given label IDs, not by their position
    label_ids = [4, 7, 9]

    # Apply mask flattening (smallest wins)
    label_frame = flatten_masks_smallest_wins(masks, label_ids)

    # check output type
    assert label_frame.dtype == np.uint16

    # check each mask region has its label ID
    assert (label_frame[masks[0]] == label_ids[0]).all()
    # mask 1 keeps its area minus the overlap with mask 0
    assert (label_frame[masks[1] & ~masks[0]] == label_ids[1]).all()
    assert (label_frame[masks[2]] == label_ids[2]).all()

    # check background is 0
    assert (label_frame[~masks.any(axis=0)] == 0).all()


def test_flatten_masks_smallest_wins_equal_areas():
    """Check equal-area masks are painted in input order: last one wins."""
    # Define an array with overlapping masks of same area
    masks = np.zeros((2, 10, 10), bool)
    masks[0, :, :6] = True  # columns 0-5, 60 pixels
    masks[1, :, 4:] = True  # columns 4-9, 60 pixels, overlaps mask 0

    # Apply flattening
    label_ids = [1, 2]
    label_frame = flatten_masks_smallest_wins(masks, label_ids)

    # columns 4 and 5 overlap: they take the label ID of the last mask
    expected = np.zeros((10, 10), np.uint16)
    expected[:, :4] = label_ids[0]
    expected[:, 4:] = label_ids[1]
    np.testing.assert_array_equal(label_frame, expected)
