"""GUI tests for annotate_crabs_orientation_manual.py.

They drive a real napari session, built on a viewer from napari's
``make_napari_viewer`` fixture, and call ``finish_session`` instead of
closing a window. napari and pytest-qt are not dev dependencies, so these
tests are skipped unless both are installed. They run in the ``test_gui``
CI job, with a virtual display.
"""

import numpy as np
import pandas as pd
import pytest
import xarray as xr
from PIL import Image

from scripts.orientation.annotate_crabs_orientation_manual import (
    STATUS_COLORS,
    build_session,
    finish_session,
)

pytest.importorskip("napari")
pytest.importorskip("pytestqt")
transform_color = pytest.importorskip(
    "napari.utils.colormaps.standardize_color"
).transform_color
Qt = pytest.importorskip("qtpy.QtCore").Qt

pytestmark = pytest.mark.gui

# Boxes per dataset and image_id, as (x centre, y centre, width, height)
# per id slot, None for an empty slot. In sep2023 image 5, boxes 0 and 1
# overlap for 24 <= x <= 30.
BOXES = {
    "aug2023": {
        0: [(20, 20, 16, 16), (60, 30, 20, 20)],
        3: [(30, 30, 20, 20), None],
    },
    "sep2023": {
        5: [(20, 20, 20, 20), (34, 20, 20, 20), (70, 40, 10, 10)],
        7: [(50, 30, 20, 20), None, None],
    },
}
IMAGE_W, IMAGE_H = 96, 64


@pytest.fixture
def package_dir(tmp_path):
    """Write a tiny labelling package with two datasets of two frames.

    ``image_id``s are sparse, and differ between datasets, so a mix-up
    between napari's frame index and ``image_id`` fails.
    """
    package_dir = tmp_path / "labelling_package_20260101_000000"
    for dataset, boxes_per_image in BOXES.items():
        image_ids = sorted(boxes_per_image)
        n_ids = len(boxes_per_image[image_ids[0]])
        position = np.full((len(image_ids), 2, n_ids), np.nan)
        shape = np.full((len(image_ids), 2, n_ids), np.nan)
        filenames = {}
        for i, image_id in enumerate(image_ids):
            for box_id, box in enumerate(boxes_per_image[image_id]):
                if box is not None:
                    position[i, :, box_id] = box[:2]
                    shape[i, :, box_id] = box[2:]
            filenames[str(image_id)] = f"{dataset}_frame_{image_id:06d}.png"

        (package_dir / dataset / "frames").mkdir(parents=True)
        for filename in filenames.values():
            Image.new("RGB", (IMAGE_W, IMAGE_H), "grey").save(
                package_dir / dataset / "frames" / filename
            )
        xr.Dataset(
            {
                "position": (("image_id", "space", "id"), position),
                "shape": (("image_id", "space", "id"), shape),
                "category": (
                    ("image_id", "id"),
                    np.ones((len(image_ids), n_ids)),
                ),
            },
            coords={
                "image_id": image_ids,
                "space": np.array(["x", "y"], dtype=object),
                "id": range(n_ids),
            },
            attrs={
                "map_image_id_to_filename": filenames,
                "labelling_package": {"source_store": f"/data/{dataset}.zarr"},
            },
        ).to_zarr(package_dir / dataset / "bboxes.zarr")
    return package_dir


def _frame(session, dataset, image_id):
    """Return the napari frame of (dataset, image_id)."""
    frames = session.frames
    match = (frames["dataset"] == dataset) & (frames["image_id"] == image_id)
    return int(np.flatnonzero(match)[0])


def _line(frame, x0, y0, x1, y1):
    """Return a napari line, in (frame, y, x)."""
    return np.array([[frame, y0, x0], [frame, y1, x1]], dtype=float)


def _set_table_id(session, row, value):
    """Type ``value`` in the ``id`` column of the features table.

    Calls the table model's ``setData``, the same path as typing in a cell.
    """
    model = session.table.table.model().sourceModel()
    id_column = next(
        col
        for col in range(model.columnCount())
        if model.headerData(col, Qt.Orientation.Horizontal) == "id"
    )
    assert model.setData(
        model.index(row, id_column), value, Qt.ItemDataRole.EditRole
    )


def _edge_colors(*statuses):
    return transform_color([STATUS_COLORS[s] for s in statuses])


def test_boxes_on_right_frames(make_napari_viewer, package_dir, tmp_path):
    """Check each frame shows the boxes of its (dataset, image_id).

    Also saves a screenshot of one frame per dataset, for a visual check.
    """
    viewer = make_napari_viewer()
    session = build_session(package_dir, "alice", viewer=viewer)

    shown = {}
    for rectangle, box_id in zip(
        session.bboxes.data, session.bboxes.features["id"], strict=True
    ):
        frame = int(rectangle[0, 0])
        y_min, x_min = rectangle[:, 1:].min(axis=0)
        y_max, x_max = rectangle[:, 1:].max(axis=0)
        shown[(frame, box_id)] = (x_min, y_min, x_max, y_max)

    expected = {}
    for dataset, boxes_per_image in BOXES.items():
        for image_id, boxes in boxes_per_image.items():
            frame = _frame(session, dataset, image_id)
            for box_id, box in enumerate(boxes):
                if box is not None:
                    x, y, w, h = box
                    expected[(frame, box_id)] = (
                        x - w / 2,
                        y - h / 2,
                        x + w / 2,
                        y + h / 2,
                    )
    assert shown == expected

    for dataset, boxes_per_image in BOXES.items():
        frame = _frame(session, dataset, min(boxes_per_image))
        viewer.dims.set_current_step(0, frame)
        viewer.screenshot(tmp_path / f"{dataset}.png", canvas_only=True)
        assert (tmp_path / f"{dataset}.png").exists()


def test_line_statuses(make_napari_viewer, package_dir, monkeypatch):
    """Check lines are linked, coloured and saved by their midpoint.

    Also checks napari warns about the ambiguous and unlinked lines.
    """
    warnings = []
    monkeypatch.setattr(
        "napari.utils.notifications.show_warning", warnings.append
    )
    session = build_session(package_dir, "alice", viewer=make_napari_viewer())
    frame = _frame(session, "sep2023", 5)
    session.lines.add_lines(
        [
            _line(frame, 12, 20, 18, 22),  # in box 0 only
            _line(frame, 25, 18, 29, 22),  # in boxes 0 and 1
            _line(frame, 50, 55, 60, 60),  # in no box
        ]
    )

    np.testing.assert_array_equal(session.lines.features["id"], [0, -1, -1])
    np.testing.assert_allclose(
        session.lines.edge_color,
        _edge_colors("linked", "ambiguous", "unlinked"),
    )
    df = pd.read_csv(session.autosave_csv)
    assert df["status"].tolist() == ["linked", "ambiguous", "unlinked"]
    assert df["dataset"].tolist() == ["sep2023"] * 3
    assert df["image_id"].tolist() == [5] * 3
    assert len(warnings) == 2
    assert warnings[0].startswith("Line 1 is in boxes [0, 1]")
    assert warnings[1].startswith("Line 2 is in no box")

    # A second line for box 0 makes both ambiguous
    session.lines.add_lines([_line(frame, 10, 15, 14, 15)])
    np.testing.assert_allclose(
        session.lines.edge_color[[0, 3]],
        _edge_colors("ambiguous", "ambiguous"),
    )
    assert len(warnings) == 3
    assert warnings[2].startswith("Line 3 is in box 0, as are lines [0]")


def test_table_edits(make_napari_viewer, package_dir):
    """Check typing a line's id relinks it, and box ids cannot change."""
    session = build_session(package_dir, "alice", viewer=make_napari_viewer())
    frame = _frame(session, "sep2023", 5)
    session.lines.add_lines([_line(frame, 25, 18, 29, 22)])  # boxes 0, 1
    assert pd.read_csv(session.autosave_csv)["status"].tolist() == [
        "ambiguous"
    ]

    # Typing the right id turns the yellow line green
    _set_table_id(session, 0, 1)
    assert session.lines.features["id"].tolist() == [1]
    np.testing.assert_allclose(
        session.lines.edge_color, _edge_colors("linked")
    )
    df = pd.read_csv(session.autosave_csv)
    assert df[["id", "status"]].values.tolist() == [[1, "linked"]]

    # Typing a box's id is undone at once
    session.viewer.layers.selection.active = session.bboxes
    _set_table_id(session, 0, 99)
    np.testing.assert_array_equal(
        session.bboxes.features["id"], session.box_ids
    )


def test_resume(make_napari_viewer, package_dir):
    """Check a resumed session gets back the lines of a crashed one."""
    first = build_session(package_dir, "alice", viewer=make_napari_viewer())
    first.lines.add_lines(
        [
            _line(_frame(first, "sep2023", 7), 45, 30, 55, 32),
            _line(_frame(first, "aug2023", 3), 25, 25, 35, 35),
            _line(_frame(first, "sep2023", 5), 25, 18, 29, 22),
        ]
    )
    _set_table_id(first, 2, 1)
    saved = pd.read_csv(first.autosave_csv)
    # no finish_session, as after a crash

    second = build_session(
        package_dir,
        "alice",
        resume_csv=first.autosave_csv,
        viewer=make_napari_viewer(),
    )
    assert second.autosave_csv == first.autosave_csv
    assert second.timestamp == first.timestamp
    assert first.autosave_csv.with_name(
        first.autosave_csv.name + ".bak"
    ).exists()
    pd.testing.assert_frame_equal(pd.read_csv(second.autosave_csv), saved)
    np.testing.assert_array_equal(second.lines.features["id"], [0, 0, 1])
    np.testing.assert_allclose(
        second.lines.edge_color, _edge_colors("linked", "linked", "linked")
    )

    # Another annotator cannot resume this CSV
    with pytest.raises(ValueError, match="same annotator"):
        build_session(package_dir, "bob", resume_csv=first.autosave_csv)


def test_results_on_close(make_napari_viewer, package_dir):
    """Check the results stores are written, overwritten and removed."""
    first = build_session(package_dir, "alice", viewer=make_napari_viewer())
    first.lines.add_lines(
        [
            _line(_frame(first, "sep2023", 5), 12, 20, 18, 26),  # green
            _line(_frame(first, "sep2023", 5), 25, 18, 29, 22),  # yellow
            _line(_frame(first, "aug2023", 3), 25, 30, 35, 30),  # green
        ]
    )
    stores = finish_session(first)
    store_name = f"orientation_alice_{first.timestamp}.zarr"
    # the labels directory is next to the package, with its timestamp
    labels_dir = package_dir.parent / "manual_labels_20260101_000000"
    assert first.autosave_csv.parent == labels_dir
    sep_store = labels_dir / "sep2023" / store_name
    aug_store = labels_dir / "aug2023" / store_name
    assert sorted(stores) == sorted([sep_store, aug_store])
    sep = xr.open_zarr(sep_store)
    assert int(sep["segment_theta"].notnull().sum()) == 1
    assert sep["segment_theta"].sel(image_id=5, id=0) == pytest.approx(
        np.pi / 4
    )
    assert sep.attrs["labelling_session"]["annotator"] == "alice"
    assert xr.open_zarr(aug_store)["segment_theta"].sel(
        image_id=3, id=0
    ) == pytest.approx(0)

    # Resume, add a green line: the same store is overwritten
    second = build_session(
        package_dir,
        "alice",
        resume_csv=first.autosave_csv,
        viewer=make_napari_viewer(),
    )
    second.lines.add_lines(
        [_line(_frame(second, "sep2023", 7), 45, 30, 55, 30)]
    )
    assert finish_session(second) == stores
    assert int(xr.open_zarr(sep_store)["segment_theta"].notnull().sum()) == 2

    # Delete the only aug2023 line: its store is removed
    second.lines.selected_data = {2}
    second.lines.remove_selected()
    assert finish_session(second) == [sep_store]
    assert not aug_store.exists()


def _table_ids(session):
    """Return the ``Index`` values of the rows the features table shows."""
    proxy = session.table.table.model()
    return sorted(
        proxy.data(proxy.index(row, 0)) for row in range(proxy.rowCount())
    )


def test_starts_drawing_lines(make_napari_viewer, package_dir):
    """Check the session starts on the lines layer, in add line mode."""
    session = build_session(package_dir, "alice", viewer=make_napari_viewer())
    assert session.viewer.layers.selection.active is session.lines
    assert session.lines.mode == "add_line"


def test_table_shows_current_frame(make_napari_viewer, package_dir):
    """Check the features table lists the current frame's shapes only."""
    session = build_session(package_dir, "alice", viewer=make_napari_viewer())
    sep5, sep7 = _frame(session, "sep2023", 5), _frame(session, "sep2023", 7)
    session.lines.add_lines(
        [
            _line(sep5, 12, 20, 18, 22),
            _line(sep7, 45, 30, 55, 32),
            _line(sep5, 66, 40, 74, 40),
        ]
    )

    session.viewer.dims.set_current_step(0, sep5)
    assert _table_ids(session) == [0, 2]
    session.viewer.dims.set_current_step(0, sep7)
    assert _table_ids(session) == [1]

    # Boxes are not listed
    session.viewer.layers.selection.active = session.bboxes
    assert _table_ids(session) == []


def test_show_panel_and_progress(make_napari_viewer, package_dir):
    """Check unticked lines and boxes are hidden, and progress is shown."""
    session = build_session(package_dir, "alice", viewer=make_napari_viewer())
    frame = _frame(session, "sep2023", 5)
    session.viewer.dims.set_current_step(0, frame)
    session.lines.add_lines(
        [
            _line(frame, 12, 20, 18, 22),  # green, for box 0
            _line(frame, 50, 55, 60, 60),  # red
        ]
    )
    assert "1 / 3 boxes labelled" in session.viewer.canvas.overlays.text.text

    session.show["linked"].setChecked(False)
    np.testing.assert_array_equal(session.lines.edge_color[:, 3], [0, 1])

    # Only box 0 of sep2023 image 5 has a green line, so by default only
    # it is hidden
    box_0 = (session.box_frames == frame) & (session.box_ids == 0)
    np.testing.assert_array_equal(session.bboxes.edge_color[:, 3], ~box_0)

    session.show["done"].setChecked(True)
    session.show["to do"].setChecked(False)
    np.testing.assert_array_equal(session.bboxes.edge_color[:, 3], box_0)
