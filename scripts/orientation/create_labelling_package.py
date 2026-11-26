"""Build a labelling package for manual crab orientation labelling.

Sample a few annotated frames per video from one or more input stores,
and pack them, with a slice of each store, into a labelling package that
``annotate_crabs_orientation_manual.py`` opens.

Each input store is a ``bboxes_and_masks_<timestamp>.zarr`` written by
``generate_masks_from_bboxes.py``, in the layout it expects:

    <data_dir>/
        frames/                                 # image files (*.png)
        annotations/
            bboxes_and_masks_<timestamp>.zarr   # input store

Output is one subdirectory per input store, named after its data
directory:

    <output_dir>/labelling_package_<timestamp>/
    ├── sep2023-full/
    │   ├── frames/          # the selected frames, copied unchanged
    │   └── bboxes.zarr      # slice of the sep2023-full store
    └── aug2023-full/
        ├── frames/
        └── bboxes.zarr

``annotate_crabs_orientation_manual.py`` writes its labels next to the
package, to ``<output_dir>/manual_labels_<timestamp>/``, with the same
timestamp.

Each ``bboxes.zarr`` is the input store sliced along ``image_id`` to the
selected frames, with the bounding box variables only (``position``,
``shape``, ``category``) and ethology's root attributes only. So it is an
ethology annotations dataset with the input store's ``image_id`` and ``id``
values, and no masks, ellipses or SAM2 metadata. Its
``map_image_id_to_filename`` attribute is cut to the selected frames, and a
``labelling_package`` attribute records how the package was built
(``source_store``, ``seed``, ``camera``, ``frames_per_video`` and
``excluded_packages``).

Frames are sampled per video, after keeping only the videos of one camera
('Right' or 'Left'). Frames used by earlier packages can be excluded with
``--exclude-package``, to label new frames in a later round. A video with
fewer frames left than ``--frames-per-video`` gives what is left, and is
reported.

Usage
From the repo root:
    uv run scripts/orientation/create_labelling_package.py \
        /path/to/sep2023-full/annotations/bboxes_and_masks_<ts>.zarr \
        /path/to/aug2023-full/annotations/bboxes_and_masks_<ts>.zarr \
        --output-dir /path/to/output_dir

    # a later round, skipping the frames of an earlier package
    uv run scripts/orientation/create_labelling_package.py \
        /path/to/sep2023-full/annotations/bboxes_and_masks_<ts>.zarr \
        /path/to/aug2023-full/annotations/bboxes_and_masks_<ts>.zarr \
        --output-dir /path/to/output_dir \
        --exclude-package /path/to/labelling_package_<ts>

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
    packages_to_exclude: Iterable[Path], ds: xr.Dataset, dataset_name: str
) -> set[int]:
    """Return the ``image_id`` values used by earlier labelling packages."""
    excluded: set[int] = set()
    for previous_package in packages_to_exclude:
        # Check if the previous package has labels for this dataset
        store = Path(previous_package) / dataset_name / "bboxes.zarr"
        if not store.exists():
            continue

        # Check if image_ids are comparable across packages
        ds_previous_package = xr.open_zarr(store)
        if (
            ds_previous_package.attrs["annotation_files"]
            != ds.attrs["annotation_files"]
        ):
            raise ValueError(
                f"{store} was built from "
                f"{ds_previous_package.attrs['annotation_files']}, "
                f"not {ds.attrs['annotation_files']}, so its image_id "
                "values may refer to a different set of frames."
            )

        # Add to unique set of image_ids
        excluded |= {int(i) for i in ds_previous_package.image_id.values}
    return excluded


def select_frames(
    ds: xr.Dataset,
    camera: str,
    frames_per_video: int,
    seed: int,
    exclude: Iterable[int] = frozenset(),
) -> np.ndarray:
    """Sample a subset of frames per video for one camera.

    A frame's video is the part of its filename before ``_frame``. A
    video belongs to ``camera`` if its name contains ``-<camera>``. Frames
    whose ``image_id`` is in ``exclude`` are not sampled. A video with
    fewer frames left than ``frames_per_video`` gives all frames that it has
    left, and this is reported.

    Returns the selected ``image_id`` values, sorted.
    """
    # build helper dataframe for frames
    df_frames = pd.DataFrame(
        sorted(_filenames(ds).items()), columns=["image_id", "filename"]
    )
    df_frames["video"] = df_frames["filename"].str.split("_frame").str[0]

    # select only rows from the relevant camera
    df_frames = df_frames[
        df_frames["video"].str.contains(f"-{camera}", regex=False)
    ]

    # list the videos before excluding frames, so that a video with no
    # frames left is still reported
    videos = sorted(df_frames["video"].unique())

    # exclude relevant image_ids
    df_frames = df_frames[~df_frames["image_id"].isin(set(exclude))]

    # loop thru videos
    selected_frames: list = []
    for video in videos:
        frames_in_video = df_frames[df_frames["video"] == video]

        # if less frames remain than requested, take all available
        # and notify
        if len(frames_in_video) < frames_per_video:
            print(
                f"{video}: {len(frames_in_video)} frames left, "
                f"{frames_per_video} requested"
                + (" (skipped)" if len(frames_in_video) == 0 else "")
            )
        n_frames = min(frames_per_video, len(frames_in_video))

        # add to list
        selected_frames.extend(
            frames_in_video.sample(n_frames, random_state=seed)["image_id"]
        )
    return np.sort(np.array(selected_frames, dtype=int))


def create_labelling_subdirectory(
    ds_bboxes: xr.Dataset,
    image_ids: np.ndarray,
    annotations_zarr: Path,
    labelling_package_dir: Path,
    labelling_package_settings: dict[str, Any],
) -> Path:
    """Create the labelling package subdirectory for one dataset.

    The labelling package has one subdirectory per dataset.
    The subdirectory is named after the annotations_zarr's grandparent
    directory (sep2023-full, aug2023-full), and the frames are copied from its
    ``frames/`` folder.

    Returns the path of the subdirectory.
    """
    # Slice the annotations store for the selected frames;
    # bounding box variables and ethology attributes only
    # (no masks, ellipses or SAM2 metadata should reach the labeller)
    frame_filenames = _filenames(ds_bboxes)
    selected_image_ids = image_ids.tolist()
    ds_slice = ds_bboxes.sel(image_id=selected_image_ids)[BBOX_VARS].load()

    if set(ds_slice.dims) != BBOX_DIMS:
        raise ValueError(
            f"Expected dimensions {BBOX_DIMS}, got {list(ds_slice.dims)}"
        )

    # zarr 3 has no stable format for numpy's fixed-width strings, so store
    # the `space` coordinate as an object
    ds_slice = ds_slice.assign_coords(space=ds_slice.space.astype(object))

    # Add relevant metadata to slice
    ds_slice.attrs = {
        **{key: ds_bboxes.attrs[key] for key in ETHOLOGY_ATTRS},
        # overrides the full map from ds: selected frames only
        "map_image_id_to_filename": {
            str(i): frame_filenames[i] for i in selected_image_ids
        },
        "labelling_package": {
            "source_store": str(Path(annotations_zarr).resolve()),
            **labelling_package_settings,
        },
    }

    # Encode with zarr's default chunks rather than the input store's
    for variable in ds_slice.variables.values():
        variable.encoding = {}

    # Write to zarr
    # (annotations_zarr is usually
    # CrabLabels/sep2023-full/annotations/bboxes_and_masks_....zarr)
    annotations_dataset_dir = Path(annotations_zarr).parent.parent
    labelling_package_dataset_dir = (
        labelling_package_dir / annotations_dataset_dir.name
    )
    ds_slice.to_zarr(labelling_package_dataset_dir / "bboxes.zarr", mode="w-")

    # Copy the selected frames unchanged
    annotation_frames_dir = annotations_dataset_dir / "frames"
    labelling_package_frames_dir = labelling_package_dataset_dir / "frames"
    labelling_package_frames_dir.mkdir(parents=True)
    for image_id in selected_image_ids:
        shutil.copy2(
            annotation_frames_dir / frame_filenames[image_id],
            labelling_package_frames_dir,
        )

    return labelling_package_dataset_dir


def main(args: argparse.Namespace) -> None:
    """Build a labelling package from the input stores."""
    # Check the input stores and subdirectory names before writing anything
    not_stores = [
        str(store)
        for store in args.stores
        if Path(store).suffix != ".zarr" or not Path(store).is_dir()
    ]
    if not_stores:
        raise ValueError(
            f"Not zarr stores: {not_stores}. Pass the output directory "
            "with --output-dir."
        )
    # dataset_names will usually be ['sep2023-full' and 'aug2023-full']
    dataset_names = [Path(store).parent.parent.name for store in args.stores]
    if len(set(dataset_names)) < len(dataset_names):
        raise ValueError(
            f"The stores' data directories must have distinct names, "
            f"got {dataset_names}"
        )

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    labelling_package_dir = args.output_dir / f"labelling_package_{timestamp}"
    labelling_package_settings = {
        "seed": args.seed,
        "camera": args.camera,
        "frames_per_video": args.frames_per_video,
        "excluded_packages": [
            str(Path(p).resolve()) for p in args.exclude_package
        ],
    }

    for bboxes_store_path, dataset_name in zip(
        args.stores, dataset_names, strict=True
    ):
        print(f"{dataset_name}: sampling frames from {bboxes_store_path}")

        ds_bboxes = load_store(bboxes_store_path)
        image_ids_to_exclude = excluded_image_ids(
            args.exclude_package,
            ds_bboxes,
            dataset_name,
        )

        image_ids = select_frames(
            ds_bboxes,
            args.camera,
            args.frames_per_video,
            args.seed,
            image_ids_to_exclude,
        )

        labelling_package_subdir = create_labelling_subdirectory(
            ds_bboxes,
            image_ids,
            bboxes_store_path,
            labelling_package_dir,
            labelling_package_settings,
        )
        print(
            f"{dataset_name}: wrote {len(image_ids)} "
            f"frames to {labelling_package_subdir}"
        )

    print(f"Labelling package written to {labelling_package_dir}")


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
            "generate_masks_from_bboxes.py."
        ),
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path.cwd(),
        help=(
            "Directory the labelling_package_<timestamp> package is "
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
            "Earlier labelling package whose frames should not be sampled "
            "again. Can be passed several times."
        ),
    )
    return parser.parse_args(list_args)


if __name__ == "__main__":
    args = parse_args(sys.argv[1:])
    main(args)
