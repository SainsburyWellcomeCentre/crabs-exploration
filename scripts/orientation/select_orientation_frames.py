"""Build a labelling package for manual crab orientation labelling.

Sample a few annotated frames per video from one or more input stores,
and pack them, with a slice of each store, into a labelling package that
``annotate_crabs_orientation_manual.py`` opens.

Each input store is a ``bboxes_and_masks_<timestamp>.zarr`` written by
``generate_masks_from_bboxes.py``, in the layout it expects::

    <data_dir>/
        frames/                                 # image files (*.png)
        annotations/
            bboxes_and_masks_<timestamp>.zarr   # input store

Output is one subdirectory per input store, named after its data
directory::

    <output_dir>/orientation_labelling_<timestamp>/
    ├── sep2023-full/
    │   ├── frames/          # the selected frames, copied unchanged
    │   └── bboxes.zarr      # slice of the sep2023-full store
    └── aug2023-full/
        ├── frames/
        └── bboxes.zarr

Each ``bboxes.zarr`` is the input store sliced along ``image_id`` to the
selected frames, with the bounding box variables only (``position``,
``shape``, ``category``) and ethology's root attributes only. So it is an
ethology annotations dataset with the input store's ``image_id`` and ``id``
values, and no masks, ellipses or SAM2 metadata. Its
``map_image_id_to_filename`` attribute is cut to the selected frames, and a
``labelling_package`` attribute records how the package was built
(``source_store``, ``seed``, ``camera``, ``frames_per_video`` and
``excluded_packages``).

Frames are sampled per video, after keeping only the videos of one camera.
Frames used by earlier packages can be excluded with ``--exclude-package``,
to label new frames in a later round. A video with fewer frames left than
``--frames-per-video`` gives what is left, and is reported.

Usage
From the repo root:
    uv run scripts/orientation/select_orientation_frames.py \
        /path/to/sep2023-full/annotations/bboxes_and_masks_<ts>.zarr \
        /path/to/aug2023-full/annotations/bboxes_and_masks_<ts>.zarr \
        --output-dir /path/to/output_dir

    # a later round, skipping the frames of an earlier package
    uv run scripts/orientation/select_orientation_frames.py \
        /path/to/sep2023-full/annotations/bboxes_and_masks_<ts>.zarr \
        /path/to/aug2023-full/annotations/bboxes_and_masks_<ts>.zarr \
        --output-dir /path/to/output_dir \
        --exclude-package /path/to/orientation_labelling_<ts>

Dependencies are auto-installed in an ephemeral environment via uv.
"""

# /// script
# requires-python = ">=3.11"
# dependencies = [
#   "numpy",
#   "pandas",
#   "xarray",
#   "zarr>=3",
#   "dask",
# ]
# ///

import argparse
import shutil
import sys
from collections.abc import Iterable
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import xarray as xr

# Variables, dimensions and root attributes of an ethology bounding boxes
# annotations dataset. The package keeps these only.
BBOX_VARS = ["position", "shape", "category"]
BBOX_DIMS = {"image_id", "space", "id"}
ETHOLOGY_ATTRS = [
    "annotation_files",
    "annotation_format",
    "images_directories",
    "map_category_to_str",
    "map_image_id_to_filename",
]


def load_store(store_path: Path) -> xr.Dataset:
    """Read an input store and check it has ethology's root attributes.

    Stores written before PR #298 have empty root attributes, so they
    have no ``map_image_id_to_filename`` to read the filenames from.
    """
    ds = xr.open_zarr(store_path)
    missing = [key for key in ETHOLOGY_ATTRS if key not in ds.attrs]
    if missing:
        raise ValueError(
            f"{store_path} has no root attributes {missing}. Write the "
            "store again with generate_masks_from_bboxes.py from the "
            "main branch."
        )
    return ds


def _filenames(ds: xr.Dataset) -> dict[int, str]:
    """Map ``image_id`` to filename, from the dataset's attributes.

    The attribute's keys are strings once read back from zarr.
    """
    return {
        int(key): value
        for key, value in ds.attrs["map_image_id_to_filename"].items()
    }


def excluded_image_ids(
    exclude_packages: Iterable[Path], ds: xr.Dataset, dataset: str
) -> set[int]:
    """Return the ``image_id`` values used by earlier labelling packages.

    ``dataset`` is the name of the package subdirectory for ``ds``. An
    earlier package without that subdirectory had no frames of this
    dataset. An earlier package built from another annotations file
    raises an error, since its ``image_id`` values may name other frames.
    """
    excluded: set[int] = set()
    for old_package in exclude_packages:
        store = Path(old_package) / dataset / "bboxes.zarr"
        if not store.exists():
            continue
        old_pkg = xr.open_zarr(store)
        if old_pkg.attrs["annotation_files"] != ds.attrs["annotation_files"]:
            raise ValueError(
                f"{store} was built from {old_pkg.attrs['annotation_files']}"
                f", not {ds.attrs['annotation_files']}, so its image_id "
                "values may name other frames."
            )
        excluded |= {int(i) for i in old_pkg.image_id.values}
    return excluded


def select_frames(
    ds: xr.Dataset,
    camera: str,
    frames_per_video: int,
    seed: int,
    exclude: Iterable[int] = frozenset(),
) -> np.ndarray:
    """Sample ``frames_per_video`` frames per video of one camera.

    A frame's video is the part of its filename before ``_frame``, and a
    video belongs to ``camera`` if its name contains ``-<camera>``. Frames
    whose ``image_id`` is in ``exclude`` are not sampled. A video with
    fewer frames left than ``frames_per_video`` gives what is left, and is
    reported.

    Returns the selected ``image_id`` values, sorted.
    """
    frames = pd.DataFrame(
        sorted(_filenames(ds).items()), columns=["image_id", "filename"]
    )
    frames["video"] = frames["filename"].str.split("_frame").str[0]
    frames = frames[frames["video"].str.contains(f"-{camera}", regex=False)]
    videos = sorted(frames["video"].unique())
    frames = frames[~frames["image_id"].isin(set(exclude))]

    selected = []
    for video in videos:
        frames_left = frames[frames["video"] == video]
        n_frames = min(frames_per_video, len(frames_left))
        if n_frames < frames_per_video:
            print(
                f"{video}: {len(frames_left)} frames left, "
                f"{frames_per_video} requested"
                + (" (skipped)" if n_frames == 0 else "")
            )
        selected.extend(
            frames_left.sample(n_frames, random_state=seed)["image_id"]
        )
    return np.sort(np.array(selected, dtype=int))


def write_package_subdir(
    ds: xr.Dataset,
    image_ids: np.ndarray,
    store_path: Path,
    package_dir: Path,
    package_settings: dict[str, Any],
) -> Path:
    """Write one package subdirectory: the selected frames and a slice.

    The subdirectory is named after the store's data directory,
    ``store_path.parent.parent``, and the frames are copied from its
    ``frames/`` folder. ``package_settings`` are the sampling settings,
    saved with the store's path in the ``labelling_package`` attribute.

    Returns the path of the subdirectory.
    """
    data_dir = Path(store_path).parent.parent
    subdir = package_dir / data_dir.name
    filenames = _filenames(ds)
    selected = [int(i) for i in image_ids]

    # Slice the store: bounding box variables and ethology's attributes
    # only, so no masks, ellipses or SAM2 metadata reach the labeller
    pkg = ds.sel(image_id=selected)[BBOX_VARS].load()
    # zarr 3 has no stable format for numpy's fixed-width strings, so store
    # the `space` coordinate as an object (as create_dataset_store does)
    pkg = pkg.assign_coords(space=pkg.space.astype(object))
    pkg.attrs = {
        **{key: ds.attrs[key] for key in ETHOLOGY_ATTRS},
        "map_image_id_to_filename": {str(i): filenames[i] for i in selected},
        "labelling_package": {
            "source_store": str(Path(store_path).resolve()),
            **package_settings,
        },
    }
    if set(pkg.data_vars) != set(BBOX_VARS) or set(pkg.dims) != BBOX_DIMS:
        raise ValueError(
            f"Expected variables {BBOX_VARS} and dimensions {BBOX_DIMS}, "
            f"got {list(pkg.data_vars)} and {list(pkg.dims)}"
        )
    # write with zarr's default chunks rather than the input store's
    for variable in pkg.variables.values():
        variable.encoding = {}

    # Copy the selected frames unchanged
    frames_dir = subdir / "frames"
    frames_dir.mkdir(parents=True)
    for image_id in selected:
        shutil.copy2(data_dir / "frames" / filenames[image_id], frames_dir)

    pkg.to_zarr(subdir / "bboxes.zarr", mode="w-")
    return subdir


def main(args: argparse.Namespace) -> None:
    """Build a labelling package from the input stores."""
    # Check the subdirectory names before writing anything
    datasets = [Path(store).parent.parent.name for store in args.stores]
    if len(set(datasets)) < len(datasets):
        raise ValueError(
            f"The stores' data directories must have distinct names, "
            f"got {datasets}"
        )

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    package_dir = args.output_dir / f"orientation_labelling_{timestamp}"
    package_settings = {
        "seed": args.seed,
        "camera": args.camera,
        "frames_per_video": args.frames_per_video,
        "excluded_packages": [
            str(Path(p).resolve()) for p in args.exclude_package
        ],
    }

    for store_path, dataset in zip(args.stores, datasets, strict=True):
        print(f"{dataset}: sampling frames from {store_path}")
        ds = load_store(store_path)
        exclude = excluded_image_ids(args.exclude_package, ds, dataset)
        image_ids = select_frames(
            ds, args.camera, args.frames_per_video, args.seed, exclude
        )
        subdir = write_package_subdir(
            ds, image_ids, store_path, package_dir, package_settings
        )
        print(f"{dataset}: wrote {len(image_ids)} frames to {subdir}")

    print(f"Labelling package written to {package_dir}")


def parse_args(list_args: list[str]) -> argparse.Namespace:
    """Parse CLI args."""
    parser = argparse.ArgumentParser(
        description=(
            "Sample frames from bboxes_and_masks_*.zarr stores and pack "
            "them, with a slice of each store, into a labelling package "
            "for manual crab orientation labelling."
        ),
    )
    parser.add_argument(
        "stores",
        type=Path,
        nargs="+",
        help=(
            "Input bboxes_and_masks_*.zarr stores, written by "
            "generate_masks_from_bboxes.py. Each store's frames are read "
            "from <store>/../../frames."
        ),
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path.cwd(),
        help=(
            "Directory the orientation_labelling_<timestamp> package is "
            "written into (default: current directory)."
        ),
    )
    parser.add_argument(
        "--camera",
        type=str,
        default="Right",
        help="Camera whose videos are sampled (default: Right).",
    )
    parser.add_argument(
        "--frames-per-video",
        type=int,
        default=2,
        help="Number of frames sampled per video (default: 2).",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for sampling the frames (default: 42).",
    )
    parser.add_argument(
        "--exclude-package",
        type=Path,
        action="append",
        default=[],
        help=(
            "Earlier labelling package whose frames are not sampled again. "
            "Can be passed several times."
        ),
    )
    return parser.parse_args(list_args)


if __name__ == "__main__":
    args = parse_args(sys.argv[1:])
    main(args)
