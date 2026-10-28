"""Manually label crab orientation in napari, one line per bounding box.

Opens a labelling package written by ``create_labelling_package.py``,
and shows the frames of all its subdirectories as one stack in napari,
with the following layers:

* ``frames``: the package's frames, read lazily;
* ``bboxes``: the annotated bounding boxes, locked, showing their ``id``;
* ``orientation lines``: the lines the labeller draws, one per box,
  parallel to the crab's long axis.

A line is linked to the box whose rectangle contains its midpoint. Its
``id`` is guessed when the line is drawn or moved, and can be typed in
the features table, which opens on start-up. Each line is coloured by
its status:

* green (``linked``): its ``id`` is a box containing its midpoint, and no
  other line on the frame has the same ``id``;
* yellow (``ambiguous``): its midpoint is in 2 or more boxes and its
  ``id`` is -1, or another line on the frame has the same ``id``;
* red (``unlinked``): its midpoint is in no box, or its ``id`` is a box
  that does not contain its midpoint.

Only green lines are saved as results.

Outputs, in ``manual_labels_<package timestamp>/``, next to the package
directory ``labelling_package_<package timestamp>/``. ``<timestamp>``
in the file names is the session's:

* ``orientation_lines_<annotator>_<timestamp>.csv``: every line, in every
  status, autosaved on every edit. Columns: ``dataset``, ``image_id``,
  ``id``, ``x0``, ``y0``, ``x1``, ``y1``, ``segment_theta``, ``status``.
  This is the file passed to ``--resume``.

* ``<dataset>/orientation_<annotator>_<timestamp>.zarr``: one store per
  package subdirectory with at least one green line, written when the
  napari window is closed. Each is the subdirectory's ``bboxes.zarr``
  plus a ``segment_theta`` (image_id, id) variable, NaN where there is no
  green line, and a ``labelling_session`` attribute.

``segment_theta`` follows the convention of ``ellipse_major_axis_theta``
in ``generate_masks_from_bboxes.py``: the angle of the line from the image
x-axis towards the y-axis, in radians, in [0, pi). The image y-axis points
down, so a positive angle looks clockwise on screen.

``--resume`` continues a session: it reads the lines from the CSV, keeps
autosaving into it (after copying it to ``<csv>.bak``), and overwrites
that session's zarr stores when the window is closed.

Usage
From the repo root:
    uv run scripts/orientation/annotate_crabs_orientation_manual.py \
        /path/to/labelling_package_<timestamp> --annotator <name>

To continue a previous session:
    uv run scripts/orientation/annotate_crabs_orientation_manual.py \
        /path/to/labelling_package_<timestamp> --annotator <name> \
        --resume /path/to/manual_labels_<ts>/orientation_lines_<name>_<ts>.csv

Dependencies are auto-installed in an ephemeral environment via uv.
"""

# /// script
# requires-python = ">=3.11"
# dependencies = [
#   # napari is pinned because the features table's `toggle` and
#   # `dataChanged` signal used below are internal to napari_builtins, and
#   # were checked in this version only. Change it together with the pin
#   # in the test_gui job of .github/workflows/test_and_deploy.yml
#   # PyQt6, as napari deprecates PyQt5 (to be removed in fall 2026)
#   "napari[pyqt6]==0.9.2",
#   "numpy",
#   "pandas",
#   "xarray",
#   "zarr>=3",
#   "dask",
#   "Pillow",
# ]
#
# [tool.uv]
# # lowest uv version tested: older ones stop with an error asking
# # to run `uv self update`
# required-version = ">=0.13.0"
# # resolve all dependencies as of the date the script was tested
# exclude-newer = "2026-10-27"
# ///

from __future__ import annotations

import argparse
import os
import re
import shutil
import sys
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import TYPE_CHECKING, Any

import dask.array as da
import numpy as np
import pandas as pd
import xarray as xr
from dask import delayed
from PIL import Image

if TYPE_CHECKING:
    import napari
    from napari.layers import Shapes
    from qtpy.QtWidgets import QWidget

# Colour of a line per status
STATUS_COLORS = {"linked": "lime", "ambiguous": "yellow", "unlinked": "red"}

# Colour of the boxes: cyan has little contrast with the frames, and
# orange is too close to the yellow and red of the lines left to fix
BOX_COLOR = "magenta"

# `id` of a line not linked to a box
UNLINKED_ID = -1

CSV_COLUMNS = [
    "dataset",
    "image_id",
    "id",
    "x0",
    "y0",
    "x1",
    "y1",
    "segment_theta",
    "status",
]

# Annotator names can be split back out of the output filenames
ANNOTATOR_PATTERN = r"[A-Za-z0-9-]+"

# Name prefixes of a package directory, and of its labels directory
PACKAGE_PREFIX = "labelling_package_"
LABELS_PREFIX = "manual_labels_"


# ---------- package -----------------
def _read_image(path: Path) -> np.ndarray:
    """Read an image file as an array."""
    return np.asarray(Image.open(path))


def load_package(
    package_dir: Path,
) -> tuple[dict[str, xr.Dataset], pd.DataFrame, da.Array]:
    """Read a labelling package and build its frame table.

    Returns:
    - the ``bboxes.zarr`` of each package subdirectory, by subdirectory
      name (the "dataset");
    - the frame table, with one row per napari frame and columns
      ``dataset``, ``image_id`` and ``filename``. Datasets are in sorted
      order, and ``image_id`` is ascending within each;
    - the frames as a lazy (frame, h, w[, c]) array, in the table's
      row order, so frame ``i`` shows the file named in row ``i``.

    """
    subdirs = sorted(
        sub for sub in package_dir.iterdir() if (sub / "bboxes.zarr").is_dir()
    )
    if not subdirs:
        raise FileNotFoundError(
            f"No <subdirectory>/bboxes.zarr found in {package_dir}. "
            "Is it a package written by create_labelling_package.py?"
        )
    packages = {
        sub.name: xr.open_zarr(sub / "bboxes.zarr").load() for sub in subdirs
    }

    # Build the frame table
    rows = []
    for dataset, pkg in packages.items():
        filenames = {
            int(key): value
            for key, value in pkg.attrs["map_image_id_to_filename"].items()
        }
        for image_id in sorted(int(i) for i in pkg.image_id.values):
            rows.append((dataset, image_id, filenames[image_id]))
    frames = pd.DataFrame(rows, columns=["dataset", "image_id", "filename"])

    # Read the frames lazily, in the table's row order
    paths = [
        package_dir / dataset / "frames" / filename
        for dataset, filename in zip(
            frames["dataset"], frames["filename"], strict=True
        )
    ]
    missing = [str(path) for path in paths if not path.exists()]
    if missing:
        raise FileNotFoundError(f"Frames missing from the package: {missing}")
    sample = _read_image(paths[0])
    images = da.stack(
        [
            da.from_delayed(
                delayed(_read_image)(path),
                shape=sample.shape,
                dtype=sample.dtype,
            )
            for path in paths
        ]
    )
    return packages, frames, images


def labels_dir_of(package_dir: Path) -> Path:
    """Return the labels directory of a package, next to it.

    ``labelling_package_<timestamp>`` gets ``manual_labels_<timestamp>``.
    A package directory with another name gets ``manual_labels_<name>``.
    """
    suffix = package_dir.name.removeprefix(PACKAGE_PREFIX)
    return package_dir.parent / f"{LABELS_PREFIX}{suffix}"


def _box_corners(
    pkg: xr.Dataset, image_id: int
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return the ``id`` and corners of the boxes of one image.

    ``position`` is the box centre in ethology, so a box spans
    ``position ± shape / 2``. Returns the ``id`` of each box, and its
    (x, y) min and max corners, as (n_boxes, 2) arrays. ``id`` slots with
    no box are left out.
    """
    image = pkg.sel(image_id=image_id, space=["x", "y"])
    position = image["position"].transpose("id", "space").values
    shape = image["shape"].transpose("id", "space").values
    has_box = ~np.isnan(position).any(axis=1)
    return (
        image["id"].values[has_box],
        position[has_box] - shape[has_box] / 2,
        position[has_box] + shape[has_box] / 2,
    )


def bboxes_to_napari_shapes(
    packages: dict[str, xr.Dataset], frames: pd.DataFrame
) -> tuple[list[np.ndarray], np.ndarray]:
    """Build napari rectangles for the boxes of every frame.

    Each rectangle is a (4, 3) array of corners in (frame, y, x), with
    ``frame`` the row of the frame table. Also returns the ``id`` of each
    rectangle's box.
    """
    rectangles, box_ids = [], []
    for frame_idx, (dataset, image_id) in enumerate(
        zip(frames["dataset"], frames["image_id"], strict=True)
    ):
        ids, xy_min, xy_max = _box_corners(packages[dataset], image_id)
        for box_id, (x0, y0), (x1, y1) in zip(
            ids, xy_min, xy_max, strict=True
        ):
            rectangles.append(
                np.array(
                    [
                        [frame_idx, y0, x0],
                        [frame_idx, y0, x1],
                        [frame_idx, y1, x1],
                        [frame_idx, y1, x0],
                    ],
                    dtype=float,
                )
            )
            box_ids.append(box_id)
    return rectangles, np.array(box_ids, dtype=int)


def boxes_containing(
    pkg: xr.Dataset, image_id: int, x: float, y: float
) -> np.ndarray:
    """Return the ``id`` of the boxes of an image that contain (x, y)."""
    ids, xy_min, xy_max = _box_corners(pkg, image_id)
    inside = (
        (xy_min[:, 0] <= x)
        & (x <= xy_max[:, 0])
        & (xy_min[:, 1] <= y)
        & (y <= xy_max[:, 1])
    )
    return ids[inside]


# ---------- lines -----------------
def segment_theta(x0: float, y0: float, x1: float, y1: float) -> float:
    """Return the angle of a line from the image x-axis, in [0, pi).

    Same convention as ``ellipse_major_axis_theta``: measured from the
    image x-axis towards the y-axis, in radians. The image y-axis points
    down, so a positive angle looks clockwise on screen. A line and its
    reverse have the same angle.
    """
    return float(np.mod(np.arctan2(y1 - y0, x1 - x0), np.pi))


def _line_endpoints(
    line: np.ndarray,
) -> tuple[int, float, float, float, float]:
    """Return the frame index and (x0, y0, x1, y1) of a napari line.

    A napari line is a (2, 3) array of end points in (frame, y, x).
    """
    (frame, y0, x0), (_, y1, x1) = line
    return int(round(frame)), x0, y0, x1, y1


def _line_boxes(
    line: np.ndarray,
    packages: dict[str, xr.Dataset],
    frames: pd.DataFrame,
) -> np.ndarray:
    """Return the ``id`` of the boxes containing a line's midpoint.

    The line's napari frame is converted to its (dataset, image_id)
    through the frame table.
    """
    frame_idx, x0, y0, x1, y1 = _line_endpoints(line)
    dataset = frames["dataset"].iloc[frame_idx]
    image_id = frames["image_id"].iloc[frame_idx]
    return boxes_containing(
        packages[dataset], image_id, (x0 + x1) / 2, (y0 + y1) / 2
    )


def guess_ids(
    layer: Shapes,
    indices: list[int],
    packages: dict[str, xr.Dataset],
    frames: pd.DataFrame,
) -> None:
    """Guess the ``id`` of drawn or moved lines from their midpoint.

    A line whose midpoint is in exactly one box gets that box's ``id``.
    Otherwise it gets -1, and napari shows a warning.
    """
    from napari.utils.notifications import show_warning

    if not indices:
        return
    line_ids = layer.features["id"].to_numpy(copy=True)
    for i in indices:
        ids = _line_boxes(layer.data[i], packages, frames)
        line_ids[i] = ids[0] if len(ids) == 1 else UNLINKED_ID
        if len(ids) == 0:
            show_warning(
                f"Line {i} is in no box, so it will not be saved: "
                "move it into a box."
            )
        elif len(ids) > 1:
            show_warning(
                f"Line {i} is in boxes {[int(box) for box in ids]}: type "
                "the right id in the features table."
            )
    # the setter emits events.features, so the features table redraws
    features = layer.features.copy()
    features["id"] = line_ids
    layer.features = features


def line_status(
    layer: Shapes,
    packages: dict[str, xr.Dataset],
    frames: pd.DataFrame,
) -> np.ndarray:
    """Return the status of every line: linked, ambiguous or unlinked."""
    line_ids = layer.features["id"].to_numpy()
    status = np.full(len(layer.data), "unlinked", dtype=object)
    for i, line in enumerate(layer.data):
        ids = _line_boxes(line, packages, frames)
        if line_ids[i] in ids:
            status[i] = "linked"
        elif line_ids[i] == UNLINKED_ID and len(ids) > 1:
            status[i] = "ambiguous"

    # Lines sharing an `id` on the same frame are ambiguous
    linked = status == "linked"
    frame_and_id = pd.DataFrame(
        {
            "frame": [_line_endpoints(line)[0] for line in layer.data],
            "id": line_ids,
        }
    )
    duplicated = frame_and_id[linked].duplicated(keep=False)
    status[np.flatnonzero(linked)[duplicated.to_numpy()]] = "ambiguous"
    return status


def lines_to_dataframe(
    layer: Shapes, status: np.ndarray, frames: pd.DataFrame
) -> pd.DataFrame:
    """Convert the lines to CSV rows, keyed by (dataset, image_id)."""
    rows = []
    for line, line_id, line_status_i in zip(
        layer.data, layer.features["id"], status, strict=True
    ):
        frame_idx, x0, y0, x1, y1 = _line_endpoints(line)
        rows.append(
            {
                "dataset": frames["dataset"].iloc[frame_idx],
                "image_id": frames["image_id"].iloc[frame_idx],
                "id": int(line_id),
                "x0": x0,
                "y0": y0,
                "x1": x1,
                "y1": y1,
                "segment_theta": segment_theta(x0, y0, x1, y1),
                "status": line_status_i,
            }
        )
    return pd.DataFrame(rows, columns=CSV_COLUMNS)


def dataframe_to_lines(
    df: pd.DataFrame, frames: pd.DataFrame
) -> tuple[list[np.ndarray], np.ndarray]:
    """Convert CSV rows back to napari lines and their ``id``.

    Each row's (dataset, image_id) is converted to its napari frame
    through the frame table.
    """
    frame_of = {
        (dataset, image_id): frame_idx
        for frame_idx, (dataset, image_id) in enumerate(
            zip(frames["dataset"], frames["image_id"], strict=True)
        )
    }
    lines = []
    for row in df.itertuples():
        frame_idx = frame_of.get((row.dataset, row.image_id))
        if frame_idx is None:
            raise ValueError(
                f"Frame ({row.dataset}, image_id {row.image_id}) is not in "
                "the labelling package. Was the CSV written for another "
                "package?"
            )
        lines.append(
            np.array(
                [[frame_idx, row.y0, row.x0], [frame_idx, row.y1, row.x1]],
                dtype=float,
            )
        )
    return lines, df["id"].to_numpy(dtype=int)


def _autosave_lines(
    layer: Shapes,
    status: np.ndarray,
    frames: pd.DataFrame,
    autosave_csv: Path,
) -> None:
    """Save the lines to CSV.

    Atomic write via tmp + os.replace, so a crash mid-write cannot
    corrupt the previous good file.
    """
    df = lines_to_dataframe(layer, status, frames)
    tmp_path = autosave_csv.with_suffix(".csv.tmp")
    df.to_csv(tmp_path, index=False)
    os.replace(tmp_path, autosave_csv)


def refresh_lines(
    layer: Shapes,
    packages: dict[str, xr.Dataset],
    frames: pd.DataFrame,
    autosave_csv: Path,
) -> None:
    """Colour the lines by status, refresh their text and autosave."""
    status = line_status(layer, packages, frames)
    if len(status):
        layer.edge_color = [STATUS_COLORS[s] for s in status]
    layer.refresh_text()
    _autosave_lines(layer, status, frames, autosave_csv)


# ---------- results -----------------
def write_orientation_zarrs(
    df: pd.DataFrame,
    packages: dict[str, xr.Dataset],
    labels_dir: Path,
    store_name: str,
    session_attrs: dict[str, Any],
) -> list[Path]:
    """Write one results store per dataset, from the lines' CSV rows.

    Each store is ``labels_dir / <dataset> / store_name``: the dataset's
    package slice plus ``segment_theta`` (image_id, id), from the green
    (``linked``) rows only and NaN elsewhere, and a ``labelling_session``
    attribute. A dataset with no green rows gets no store, and a store
    left by an earlier session is removed.

    Returns the paths of the stores written.
    """
    written = []
    for dataset, pkg in packages.items():
        rows = df[(df["dataset"] == dataset) & (df["status"] == "linked")]
        store = labels_dir / dataset / store_name
        if rows.empty:
            if store.exists():
                shutil.rmtree(store)
            continue

        # Place each angle at its (image_id, id)
        image_idx = pkg.indexes["image_id"].get_indexer(rows["image_id"])
        id_idx = pkg.indexes["id"].get_indexer(rows["id"])
        if (image_idx < 0).any() or (id_idx < 0).any():
            raise ValueError(
                f"Some {dataset} rows have an image_id or id that is not in "
                "the package."
            )
        theta = np.full((pkg.sizes["image_id"], pkg.sizes["id"]), np.nan)
        theta[image_idx, id_idx] = rows["segment_theta"]

        out = pkg.assign(segment_theta=(("image_id", "id"), theta))
        out.attrs = {**pkg.attrs, "labelling_session": session_attrs}
        store.parent.mkdir(parents=True, exist_ok=True)
        out.to_zarr(store, mode="w")  # overwrite on resume
        written.append(store)
    return written


# ---------- session -----------------
@dataclass
class Session:
    """What the napari event handlers and ``finish_session`` share."""

    viewer: napari.Viewer
    lines: Shapes  # the "orientation lines" layer
    bboxes: Shapes
    table: QWidget  # the features table widget
    packages: dict[str, xr.Dataset]
    frames: pd.DataFrame  # the frame table
    box_ids: np.ndarray  # original bboxes "id" feature, to undo table edits
    annotator: str
    timestamp: str  # new, or reused from the resumed CSV's name
    package_dir: Path
    autosave_csv: Path
    labels_dir: Path


def _check_annotator(annotator: str) -> str:
    """Check the annotator name can be split back out of filenames."""
    if not re.fullmatch(ANNOTATOR_PATTERN, annotator):
        raise ValueError(
            f"Annotator {annotator!r} must only have letters, digits and "
            "'-' (no spaces or '_')."
        )
    return annotator


def _resumed_timestamp(resume_csv: Path, annotator: str) -> str:
    """Return the timestamp of a CSV to resume, checking its annotator."""
    match = re.fullmatch(
        rf"orientation_lines_{annotator}_(\d{{8}}_\d{{6}})\.csv",
        resume_csv.name,
    )
    if match is None:
        raise ValueError(
            f"{resume_csv.name} is not an orientation_lines_{annotator}_"
            "<timestamp>.csv file. Resume a CSV of the same annotator."
        )
    return match.group(1)


def _start_drawing(viewer: napari.Viewer, lines: Shapes) -> None:
    """Select the lines layer, in "add line" mode."""
    viewer.layers.selection.active = lines
    lines.mode = "add_line"


def build_session(
    package_dir: Path,
    annotator: str,
    resume_csv: Path | None = None,
    viewer: napari.Viewer | None = None,
) -> Session:
    """Set up a labelling session in napari, without the event loop.

    Load the package, add the layers, add the resumed lines, open the
    features table and connect the event handlers.
    """
    import napari

    _check_annotator(annotator)
    package_dir = Path(package_dir).resolve()
    packages, frames, images = load_package(package_dir)

    labels_dir = labels_dir_of(package_dir)
    if resume_csv is not None:
        resume_csv = Path(resume_csv).resolve()
        timestamp = _resumed_timestamp(resume_csv, annotator)
        autosave_csv = resume_csv
        # so a bad resumed session (e.g. lines deleted by mistake) can be
        # undone
        shutil.copy2(
            resume_csv, resume_csv.with_name(resume_csv.name + ".bak")
        )
        lines_data, line_ids = dataframe_to_lines(
            pd.read_csv(resume_csv), frames
        )
    else:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        labels_dir.mkdir(exist_ok=True)
        autosave_csv = (
            labels_dir / f"orientation_lines_{annotator}_{timestamp}.csv"
        )
        lines_data, line_ids = [], np.array([], dtype=int)

    # Add layers
    viewer = viewer or napari.Viewer()
    viewer.add_image(images, name="frames", rgb=images.ndim == 4)
    box_shapes, box_ids = bboxes_to_napari_shapes(packages, frames)
    bboxes = viewer.add_shapes(
        box_shapes,
        ndim=3,
        shape_type="rectangle",
        edge_color=BOX_COLOR,
        face_color="transparent",
        edge_width=3,
        features={"id": box_ids},
        text={"string": "{id}", "color": BOX_COLOR, "anchor": "upper_left"},
        name="bboxes",
    )
    bboxes.editable = False
    lines = viewer.add_shapes(
        lines_data,
        ndim=3,
        shape_type="line",
        edge_color="white",
        edge_width=5,
        features={"id": line_ids},
        feature_defaults={"id": UNLINKED_ID},
        text={"string": "{id}", "color": "white", "anchor": "center"},
        name="orientation lines",
    )
    viewer.dims.set_current_step(0, 0)

    # Open the features table, with editing switched on
    _, table = viewer.window.add_plugin_dock_widget(
        "napari", "Features table widget"
    )
    table.toggle.setChecked(True)  # the "editable" switch

    # Draw lines, once the widgets are added
    _start_drawing(viewer, lines)

    session = Session(
        viewer=viewer,
        lines=lines,
        bboxes=bboxes,
        table=table,
        packages=packages,
        frames=frames,
        box_ids=box_ids,
        annotator=annotator,
        timestamp=timestamp,
        package_dir=package_dir,
        autosave_csv=autosave_csv,
        labels_dir=labels_dir,
    )

    # Number of lines before the last addition: drawing a line, or adding
    # several in code, emits a single ADDED event for all of them
    n_lines_before_adding = len(lines.data)

    def on_lines_data(event):
        nonlocal n_lines_before_adding
        if event.action == "adding":
            n_lines_before_adding = len(session.lines.data)
            return
        if event.action == "added":
            new = range(n_lines_before_adding, len(session.lines.data))
            guess_ids(session.lines, list(new), packages, frames)
        elif event.action == "changed":
            guess_ids(
                session.lines, list(event.data_indices), packages, frames
            )
        elif event.action != "removed":
            return  # skip the "changing" and "removing" events
        refresh_lines(session.lines, packages, frames, session.autosave_csv)

    def on_table_edit(*_):
        # editing the table updates layer.features but emits no layer
        # event, so listen to the table itself. Undo edits to box ids.
        if not np.array_equal(session.bboxes.features["id"], session.box_ids):
            session.bboxes.features = {"id": session.box_ids}
            session.bboxes.refresh_text()
        refresh_lines(session.lines, packages, frames, session.autosave_csv)

    session.lines.events.data.connect(on_lines_data)
    # connected after the table's own handler, which writes the edit to
    # layer.features, so the features are up to date when this one runs
    session.table.table.model().sourceModel().dataChanged.connect(
        on_table_edit
    )

    # colour resumed lines, and write the first autosave
    refresh_lines(lines, packages, frames, autosave_csv)
    print(f"Autosaving orientation lines to {autosave_csv}")
    return session


def finish_session(session: Session) -> list[Path]:
    """End a session: autosave once more, then write the results.

    The layer is not recoloured: the viewer may already be closed.
    """
    import napari

    status = line_status(session.lines, session.packages, session.frames)
    _autosave_lines(
        session.lines, status, session.frames, session.autosave_csv
    )
    session_attrs = {
        "annotator": session.annotator,
        "package_dir": str(session.package_dir),
        "lines_csv": str(session.autosave_csv),
        "napari_version": napari.__version__,
    }
    return write_orientation_zarrs(
        pd.read_csv(session.autosave_csv),
        session.packages,
        session.labels_dir,
        f"orientation_{session.annotator}_{session.timestamp}.zarr",
        session_attrs,
    )


def main(args: argparse.Namespace) -> None:
    """Launch napari to label crab orientation, then write the results."""
    import napari
    from qtpy.QtCore import QTimer

    session = build_session(args.package_dir, args.annotator, args.resume)
    # once the window is up, in case it changed the mode (one labeller's
    # first run started in "add path" mode)
    QTimer.singleShot(0, lambda: _start_drawing(session.viewer, session.lines))
    napari.run()  # blocks until the window is closed
    written = finish_session(session)
    for store in written:
        print(f"Saved orientation labels to {store}")
    if not written:
        print("No green lines: no orientation labels saved.")


def parse_args(list_args: list[str]) -> argparse.Namespace:
    """Parse CLI args."""
    parser = argparse.ArgumentParser(
        description=(
            "Label crab orientation in napari, drawing one line per "
            "bounding box of a labelling package."
        ),
    )
    parser.add_argument(
        "package_dir",
        type=Path,
        help=(
            "Labelling package directory (labelling_package_<timestamp>), "
            "written by create_labelling_package.py. Outputs are written "
            "to manual_labels_<timestamp>, next to it."
        ),
    )
    parser.add_argument(
        "--annotator",
        type=str,
        required=True,
        help=(
            "Your name, with letters, digits and '-' only. Use the same "
            "name in every session."
        ),
    )
    parser.add_argument(
        "--resume",
        type=Path,
        default=None,
        help=(
            "orientation_lines_<annotator>_<timestamp>.csv of a previous "
            "session, to continue it."
        ),
    )
    args = parser.parse_args(list_args)
    try:
        _check_annotator(args.annotator)
    except ValueError as error:
        parser.error(str(error))
    return args


if __name__ == "__main__":
    args = parse_args(sys.argv[1:])
    main(args)
