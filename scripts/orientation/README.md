## Label crabs' orientation

1. `generate_masks_from_bboxes.py`: script to generate SAM2 masks and ellipse parameters from manually annotated boxes.
2. `create_labelling_package.py`: script to build a package to label the orientation of each crab.
3. `annotate_crabs_orientation_manual.py`: script to set up napari to label crabs' orientation.

For details on how to run Steps 2 and 3, see [guides/ManualAnnotationOrientation.md](../../guides/ManualAnnotationOrientation.md).

## Developer notes

The labelling package is built from the `bboxes_and_masks_<timestamp>.zarr`
stores written by
[generate_masks_from_bboxes.py](../scripts/orientation/generate_masks_from_bboxes.py).

### Build a labelling package

To build a labelling package, we run `create_labelling_package.py`

By default, it samples 2 frames per `Right` video with seed 42:
* 28 frames from 14 labelled `Right` videos in `sep2023-full`
* 10 frames from 5 labelled `Right` videos in `aug2023-full`.

It writes
`labelling_package_<timestamp>/` under the selected output directory. It has one subfolder per store, holding
the selected frames and a slice of the store with the boxes only.

The
`labelling_package` attribute of each `bboxes.zarr` records the source
store and the sampling settings.

### Label more frames later

To build a later round's package with new frames only, pass each earlier
package with `--exclude-package`:

```bash
uv run scripts/orientation/create_labelling_package.py <sep store> <aug store> \
    --output-dir <output dir> \
    --exclude-package <first package> \
    --exclude-package <second package>
```

A video with fewer frames left than `--frames-per-video` gives what is
left, and is reported.

### Pin the labellers' version

The labellers' command runs the script from a raw GitHub URL pinned to a
tag, so everybody, in every session, runs the same code even if `main`
changes. Before sending this guide out, create the tag on the merged commit
and push it:

```bash
git tag orientation-labelling-v1 <merged commit>
git push origin orientation-labelling-v1
```

If the script needs a fix during labelling, create a new tag (for example
`orientation-labelling-v2`) and update the commands in this guide.

The script cannot know which tag it runs from, so it prints the SHA-256 of
its source at startup and records it as `script_sha256` in the
`labelling_session` attribute of each results store. To check which tag
produced a store, compare it with:

```bash
git show orientation-labelling-v1:scripts/orientation/annotate_crabs_orientation_manual.py | shasum -a 256
```

The napari version, and its Qt backend (PyQt6: napari deprecates PyQt5),
are pinned in the script's `uv` header and in the `test_gui` CI job:
change both together.
