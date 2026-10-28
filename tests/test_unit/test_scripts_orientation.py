from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import xarray as xr
from PIL import Image

from scripts.orientation.annotate_crabs_orientation_manual import (
    boxes_containing,
    boxes_done,
    labels_dir_of,
    progress_text,
    segment_theta,
    write_orientation_zarrs,
)
from scripts.orientation.create_labelling_package import (
    BBOX_VARS,
    ETHOLOGY_ATTRS,
    excluded_image_ids,
    load_store,
    main,
    parse_args,
    select_frames,
    write_package_subdir,
)

# -------- tests for create_labelling_package --------------

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


@pytest.mark.parametrize("not_store", ["output_dir", "missing.zarr"])
def test_main_rejects_non_store_paths(input_store, tmp_path, not_store):
    """Check a path that is not a zarr store fails before writing anything.

    A directory is what is passed when ``--output-dir`` is forgotten.
    """
    (tmp_path / "output_dir").mkdir()
    output_dir = tmp_path / "out"
    args = parse_args(
        [
            str(input_store),
            str(tmp_path / not_store),
            "--output-dir",
            str(output_dir),
        ]
    )
    with pytest.raises(ValueError, match="Not zarr stores"):
        main(args)
    assert not output_dir.exists()


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


# -------- tests for annotate_crabs_orientation_manual --------------


@pytest.fixture
def package_slice() -> xr.Dataset:
    """Get a package slice with sparse ``image_id``s and overlapping boxes.

    ``image_id`` is [5, 7], so a mix-up between ``image_id`` and position
    along ``image_id`` fails. In image 5, boxes 0 and 1 overlap for
    9 <= x <= 15, and ``id`` 2 has no box. Image 7 has one box.
    """
    position = np.full((2, 2, 3), np.nan)  # (image_id, space, id)
    shape = np.full((2, 2, 3), np.nan)
    position[0, :, 0], shape[0, :, 0] = [10, 10], [10, 10]  # x: 5 to 15
    position[0, :, 1], shape[0, :, 1] = [14, 10], [10, 10]  # x: 9 to 19
    position[1, :, 0], shape[1, :, 0] = [50, 50], [20, 20]
    return xr.Dataset(
        {
            "position": (("image_id", "space", "id"), position),
            "shape": (("image_id", "space", "id"), shape),
            "category": (("image_id", "id"), np.ones((2, 3))),
        },
        coords={
            "image_id": [5, 7],
            "space": np.array(["x", "y"], dtype=object),
            "id": range(3),
        },
        attrs={"map_image_id_to_filename": {"5": "a.png", "7": "b.png"}},
    )


@pytest.mark.parametrize(
    "image_id, x, y, expected_ids",
    [
        (5, 6, 10, [0]),  # one box
        (5, 12, 10, [0, 1]),  # two overlapping boxes
        (5, 30, 30, []),  # no box
        (7, 6, 10, []),  # image 5's box, on image 7
        (7, 45, 55, [0]),
    ],
)
def test_boxes_containing(package_slice, image_id, x, y, expected_ids):
    """Check the boxes containing a point are found on the right image."""
    np.testing.assert_array_equal(
        boxes_containing(package_slice, image_id, x, y), expected_ids
    )


@pytest.mark.parametrize(
    "x0, y0, x1, y1, expected_theta",
    [
        (0, 0, 10, 0, 0),  # horizontal
        (10, 0, 0, 0, 0),  # horizontal, reversed
        (0, 0, 10, 10, np.pi / 4),  # down-right on screen
        (10, 10, 0, 0, np.pi / 4),  # down-right, reversed
        (0, 0, 0, 10, np.pi / 2),  # vertical
        (0, 10, 10, 0, 3 * np.pi / 4),  # up-right on screen
    ],
)
def test_segment_theta(x0, y0, x1, y1, expected_theta):
    """Check the angle convention matches ``ellipse_major_axis_theta``.

    The angle is in [0, pi), and does not depend on the line's direction.
    """
    assert segment_theta(x0, y0, x1, y1) == pytest.approx(expected_theta)


def test_write_orientation_zarrs(package_slice, tmp_path):
    """Check green rows land at their (image_id, id), and only them.

    Yellow and red rows are not written, other cells are NaN, and a
    dataset with no green rows has its store from an earlier session
    removed.
    """
    packages = {"aug": package_slice, "sep": package_slice}
    rows = [
        ("sep", 7, 0, 0.5, "linked"),
        ("sep", 5, 1, 0.2, "linked"),
        ("sep", 5, -1, 0.9, "ambiguous"),
        ("sep", 5, 0, 1.1, "unlinked"),
        ("aug", 5, 0, 1.3, "unlinked"),
    ]
    df = pd.DataFrame(
        rows, columns=["dataset", "image_id", "id", "segment_theta", "status"]
    )
    store_name = "orientation_alice_20260101_000000.zarr"
    stale_store = tmp_path / "aug" / store_name
    package_slice.to_zarr(stale_store)

    written = write_orientation_zarrs(
        df, packages, tmp_path, store_name, {"annotator": "alice"}
    )

    assert written == [tmp_path / "sep" / store_name]
    assert not stale_store.exists()

    out = xr.open_zarr(written[0])
    expected = np.full((2, 3), np.nan)
    expected[1, 0] = 0.5  # image_id 7, id 0
    expected[0, 1] = 0.2  # image_id 5, id 1
    np.testing.assert_array_equal(out["segment_theta"], expected)
    np.testing.assert_array_equal(out.image_id, [5, 7])
    xr.testing.assert_equal(out["position"], package_slice["position"])
    assert out.attrs["labelling_session"] == {"annotator": "alice"}
    assert out.attrs["map_image_id_to_filename"] == {
        "5": "a.png",
        "7": "b.png",
    }


@pytest.mark.parametrize(
    "package_name, expected_labels_name",
    [
        ("labelling_package_20260101_000000", "manual_labels_20260101_000000"),
        ("my_package", "manual_labels_my_package"),  # renamed package
    ],
)
def test_labels_dir_of(tmp_path, package_name, expected_labels_name):
    """Check the labels directory is next to the package."""
    assert labels_dir_of(tmp_path / package_name) == (
        tmp_path / expected_labels_name
    )


def test_boxes_done():
    """Check a box is done only with a green line of its id, on its frame.

    Boxes are (frame, id) (0, 0), (0, 1), (1, 0) and (1, 1).
    """
    done = boxes_done(
        box_frames=np.array([0, 0, 1, 1]),
        box_ids=np.array([0, 1, 0, 1]),
        line_frames=np.array([0, 0, 1, 1]),
        line_ids=np.array([0, 1, 1, -1]),
        status=np.array(
            ["linked", "ambiguous", "linked", "unlinked"], dtype=object
        ),
    )
    np.testing.assert_array_equal(done, [True, False, False, True])


def test_progress_text():
    """Check the counts of boxes and frames done.

    Frame 1 has no boxes, so it is done.
    """
    frames = pd.DataFrame(
        {"dataset": ["aug", "aug", "sep"], "image_id": [3, 4, 5]}
    )
    text = progress_text(
        frame_idx=2,
        frames=frames,
        box_frames=np.array([0, 0, 2, 2, 2]),
        done=np.array([True, True, True, False, False]),
    )
    assert text == (
        "Frame 3 of 3 (sep, image_id 5)\n"
        "This frame: 1 of 3 boxes have a green line\n"
        "All frames: 3 of 5 boxes, 2 of 3 frames done"
    )
