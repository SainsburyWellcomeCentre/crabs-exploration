from pathlib import Path

import numpy as np
import pytest
import xarray as xr
from PIL import Image

from scripts.orientation.select_orientation_frames import (
    BBOX_VARS,
    ETHOLOGY_ATTRS,
    excluded_image_ids,
    load_store,
    select_frames,
    write_package_subdir,
)

# -------- tests for select_orientation_frames --------------

# Number of annotated frames per video in the synthetic store
FRAMES_PER_VIDEO_IN_STORE = {
    "01.01.2023-01-Left": 3,
    "01.01.2023-01-Right": 3,
    "01.01.2023-02-Right_RE": 5,
}


@pytest.fixture
def input_store(tmp_path) -> Path:
    """Write a synthetic input store and its frames to a data directory.

    Mimics the output of ``generate_masks_from_bboxes.py``: an ethology
    annotations dataset expanded with mask variables and SAM2 metadata,
    in ``<data_dir>/annotations/``, with its frames in
    ``<data_dir>/frames/``. ``image_id`` is the position of the filename
    in the sorted list of filenames, as in ethology.
    """
    data_dir = tmp_path / "jan2023-full"
    (data_dir / "frames").mkdir(parents=True)
    (data_dir / "annotations").mkdir()

    filenames = sorted(
        f"{video}_frame_{k:06d}.png"
        for video, n_frames in FRAMES_PER_VIDEO_IN_STORE.items()
        for k in range(n_frames)
    )
    for filename in filenames:
        Image.new("RGB", (8, 6)).save(data_dir / "frames" / filename)

    n_images, n_ids = len(filenames), 3
    rng = np.random.default_rng(0)
    ds = xr.Dataset(
        {
            "position": (
                ("image_id", "space", "id"),
                rng.uniform(size=(n_images, 2, n_ids)),
            ),
            "shape": (
                ("image_id", "space", "id"),
                rng.uniform(size=(n_images, 2, n_ids)),
            ),
            "category": (("image_id", "id"), np.ones((n_images, n_ids))),
            "labels": (
                ("image_id", "img_h", "img_w"),
                np.zeros((n_images, 6, 8), dtype=np.uint16),
            ),
            "label_of": ("id", np.arange(n_ids, dtype=np.uint16) + 1),
            "ellipse_major_axis_theta": (
                ("image_id", "id"),
                rng.uniform(size=(n_images, n_ids)),
            ),
        },
        coords={
            "image_id": range(n_images),
            "space": np.array(["x", "y"], dtype=object),
            "id": range(n_ids),
        },
        attrs={
            "annotation_files": str(data_dir / "annotations" / "coco.json"),
            "annotation_format": "COCO",
            "images_directories": str(data_dir / "frames"),
            "map_category_to_str": {"1": "crab"},
            "map_image_id_to_filename": {
                str(i): f for i, f in enumerate(filenames)
            },
            "sam2_model": "facebook/sam2.1-hiera-base-plus",
            "occlusion_policy": "smallest_wins",
        },
    )
    store_path = data_dir / "annotations" / "bboxes_and_masks_test.zarr"
    ds.to_zarr(store_path)
    return store_path


def _videos(ds: xr.Dataset, image_ids) -> list[str]:
    """Return the video of each ``image_id``, from the filenames."""
    filenames = ds.attrs["map_image_id_to_filename"]
    return [filenames[str(i)].split("_frame")[0] for i in image_ids]


def _build_package(
    store_path: Path,
    package_dir: Path,
    exclude_packages: list[Path],
    frames_per_video: int = 2,
) -> np.ndarray:
    """Build a package as ``main`` does, and return its ``image_id``s."""
    ds = load_store(store_path)
    exclude = excluded_image_ids(exclude_packages, ds, "jan2023-full")
    image_ids = select_frames(ds, "Right", frames_per_video, 42, exclude)
    write_package_subdir(ds, image_ids, store_path, package_dir, {})
    return image_ids


def test_select_frames_and_write_package(input_store, tmp_path):
    """Check the frames sampled, and the slice written to the package.

    Only the chosen camera's videos are sampled, each gives
    ``frames_per_video`` frames, the same seed gives the same frames, and
    ``bboxes.zarr`` holds bounding box data and ethology's attributes
    only, with the input store's ``image_id`` and ``id`` values.
    """
    ds = load_store(input_store)

    # Sample frames
    image_ids = select_frames(ds, "Right", 2, seed=42)
    videos = _videos(ds, image_ids)
    assert sorted(set(videos)) == [
        "01.01.2023-01-Right",
        "01.01.2023-02-Right_RE",
    ]
    assert all(videos.count(v) == 2 for v in set(videos))
    np.testing.assert_array_equal(
        image_ids, select_frames(ds, "Right", 2, seed=42)
    )

    # Write the package subdirectory
    settings = {"seed": 42, "camera": "Right", "frames_per_video": 2}
    subdir = write_package_subdir(
        ds, image_ids, input_store, tmp_path / "package", settings
    )
    assert subdir == tmp_path / "package" / "jan2023-full"

    pkg = xr.open_zarr(subdir / "bboxes.zarr")
    assert set(pkg.data_vars) == set(BBOX_VARS)
    assert set(pkg.dims) == {"image_id", "space", "id"}
    np.testing.assert_array_equal(pkg.image_id, image_ids)
    np.testing.assert_array_equal(pkg.id, ds.id)
    for var in BBOX_VARS:
        xr.testing.assert_equal(pkg[var], ds[var].sel(image_id=image_ids))

    assert set(pkg.attrs) == {*ETHOLOGY_ATTRS, "labelling_package"}
    assert pkg.attrs["labelling_package"] == {
        "source_store": str(input_store.resolve()),
        **settings,
    }
    filenames = pkg.attrs["map_image_id_to_filename"]
    assert sorted(int(i) for i in filenames) == list(image_ids)
    assert sorted(p.name for p in (subdir / "frames").iterdir()) == sorted(
        filenames.values()
    )


def test_load_store_without_attrs(input_store):
    """Check a store with empty root attributes raises an error."""
    ds = xr.open_zarr(input_store)
    ds.attrs = {}
    ds.to_zarr(input_store, mode="w")
    with pytest.raises(ValueError, match="no root attributes"):
        load_store(input_store)


def test_exclude_earlier_packages(input_store, tmp_path, capsys):
    """Check later packages skip the frames of earlier ones.

    The Right videos have 3 and 5 frames. With 2 frames per video, the
    second round has 1 and 3 frames left, so the 3-frame video gives 1
    frame; the third round, excluding both, has 0 and 1 left.
    """
    first, second, third = (tmp_path / f"package_{i}" for i in range(3))
    ids_first = _build_package(input_store, first, [])
    ids_second = _build_package(input_store, second, [first])
    ids_third = _build_package(input_store, third, [first, second])

    assert not set(ids_first) & set(ids_second)
    assert not set(ids_third) & (set(ids_first) | set(ids_second))

    ds = load_store(input_store)
    assert sorted(_videos(ds, ids_second)) == [
        "01.01.2023-01-Right",
        "01.01.2023-02-Right_RE",
        "01.01.2023-02-Right_RE",
    ]
    assert _videos(ds, ids_third) == ["01.01.2023-02-Right_RE"]

    # Videos running short are reported
    captured = capsys.readouterr().out
    assert "01.01.2023-01-Right: 1 frames left, 2 requested\n" in captured
    assert "01.01.2023-01-Right: 0 frames left, 2 requested (skipped)" in (
        captured
    )

    # An earlier package without this dataset excludes nothing
    assert excluded_image_ids([tmp_path / "other"], ds, "jan2023-full") == (
        set()
    )

    # An earlier package from another annotations file raises an error
    ds.attrs["annotation_files"] = "/another/coco.json"
    with pytest.raises(ValueError, match="may name other frames"):
        excluded_image_ids([first], ds, "jan2023-full")
