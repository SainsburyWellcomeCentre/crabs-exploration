# Manual labelling of crab orientation

In this task you draw one line over each annotated crab, along its long axis
(left to right across its body). We use the angle of each line to check
the orientation we compute automatically for each crab.

You need about 1 GB of disk space, and no Python installation. The steps
below are for Windows, but the same commands work on macOS and Linux in a
terminal.

## 1. Install uv

`uv` runs the labelling tool, and downloads everything it needs, Python
included. Open **PowerShell** and run:

```powershell
powershell -ExecutionPolicy ByPass -c "irm https://astral.sh/uv/install.ps1 | iex"
```

Then close PowerShell and open it again, so it finds `uv`. To check it
worked, run `uv --version`.

If `uv` is too old for the tool, the tool stops with a message asking you
to run `uv self update`. Run it, then try again.

## 2. Copy the labelling package

Copy the labelling package folder from ceph to your computer, for example
into your `Documents` folder. We will tell you its location on ceph. It is
called `labelling_package_<timestamp>` and is about 0.6 GB. Your labels
will be saved next to it, in a new `manual_labels_<timestamp>` folder.

It holds two subfolders, `sep2023-full` and `aug2023-full`, each with the
frames to label and the boxes drawn on them. Do not change anything inside
it.

## 3. Start the labelling tool

In PowerShell, run the command below, with:

- the path to your copy of the package, in place of `<package>` (in File
  Explorer, Shift + right-click the folder and choose **Copy as path**);
- your name in place of `<name>`, with letters, digits and `-` only (no
  spaces or `_`), for example `jane-doe`. **Use the same name every time.**

```powershell
uv run https://raw.githubusercontent.com/SainsburyWellcomeCentre/crabs-exploration/orientation-labelling-v1/scripts/orientation/annotate_crabs_orientation_manual.py <package> --annotator <name>
```

Copy the command exactly as given: it points to a fixed version of the tool
(`orientation-labelling-v1`), so everybody labels with the same code.

The first run takes a few minutes, while `uv` downloads what the tool
needs. Later runs start in seconds.

A napari window opens with:

- the frames of both datasets, one after the other. Move between frames with
  the slider under the image;
- the boxes, in magenta, each with its `id` in the top-left corner. You
  cannot edit them;
- an empty `orientation lines` layer, selected and ready to draw lines;
- your progress, in the top-left corner of the image: how many boxes of
  this frame, and of all frames, have a green line;
- the **features table**, on the right, listing the lines of the frame you
  are on;
- the **show** panel, on the right, to hide lines and boxes.

Both are explained in [Find the work left](#find-the-work-left).

## 4. Label

For every box on every frame, draw one line **parallel to the long axis of
the crab**. To draw a line, press and drag over the crab, then release.

- Only the line's angle is used: it does not need to be centred on the crab,
  or as long as the crab.
- A line belongs to the box that contains **its midpoint**. So draw it
  over the crab, with its middle on the crab.
- Zoom with the mouse wheel. To move around the image while drawing, hold
  `Space` and drag.
- To see a crab better, hide the boxes: select the `bboxes` layer in the
  layer list and press `V`; press `V` again to show them. `Shift + V` hides
  all the other layers instead. Select `orientation lines` again before you
  draw.

Each line shows its `id` and is coloured by its status:

| Colour | Meaning | What to do |
|---|---|---|
| green | the line is linked to the box with the `id` shown | nothing |
| yellow | the midpoint is inside 2 or more boxes, so the `id` is `-1`; or another line on the frame has the same `id` | type the right `id` in the features table, or move the line |
| red | the midpoint is inside no box, or the `id` typed is a box that does not contain the midpoint | move the line into the box |

A message appears in the bottom-right corner when a new line is yellow or
red.

**Only green lines are saved as results.** Before you finish, check that
every line is green.

### Find the work left

The text in the top-left corner of the image shows, for the frame you are
on and for all frames, how many boxes have a green line, and how many
frames are done (all their boxes have a green line). Boxes you leave
without a line on purpose are counted as not done, so the counts may not
reach the totals.

To see what is left, untick options in the **show** panel:

- untick **green lines** to see only the lines left to fix;
- untick **boxes with a green line** to see only the boxes left to label.

Hidden lines can still be selected and moved, so tick them again before
you edit lines near them.

### Fix a yellow line in the features table

1. Make sure the `orientation lines` layer is selected in the layer list.
2. In the features table, find the line's row. The table only lists the
   lines of the frame you are on. The `Index` column is the line number
   given in the message.
3. Double-click its `id` cell, type the `id` of the right box, and press
   `Enter`.

The line turns green. The table lists no boxes when the `bboxes` layer is
selected: box `id`s cannot be edited.

### Move or delete a line

- Press `S` (select mode), click the line and drag it. Press `L` to go back
  to drawing lines.
- **Moving a line guesses its `id` again** from its new position, and
  replaces any `id` you typed. Type it again if needed.
- To delete a line, select it and press `Delete`.

## 5. Your files

Your work is saved automatically after every change, in a
`manual_labels_<timestamp>` folder next to the package, with the
package's timestamp. The `<timestamp>` in the file names is when you
started labelling:

```text
labelling_package_<timestamp>/        (the package: do not change it)
manual_labels_<timestamp>/
├── orientation_lines_<name>_<timestamp>.csv
├── orientation_lines_<name>_<timestamp>.csv.bak    (only after a resume)
├── sep2023-full/orientation_<name>_<timestamp>.zarr
└── aug2023-full/orientation_<name>_<timestamp>.zarr
```

- `orientation_lines_<name>_<timestamp>.csv`: your work in progress. It is
  saved after every change, and holds all your lines, red and yellow ones
  included. This is the file you pass to `--resume` (step 6).
- `orientation_lines_<name>_<timestamp>.csv.bak`: a copy of the CSV from
  before your last resume, in case lines were deleted by mistake.
- `sep2023-full/…zarr` and `aug2023-full/…zarr`: the results, from green
  lines only. They are written when you close the napari window, and are
  rebuilt from the CSV each time. If one is missing or looks wrong, resume
  and close the window again.

Do not edit, rename or move these files.

You never need napari's **File > Save** (`Ctrl + S`): your work is saved
automatically. That menu exports layers in napari's own format, which is
not what we need.

## 6. Stop and resume

To stop, close the napari window. This writes the results.

To continue later, or after a crash, run the same command as in step 3, with
`--resume` and the path to your CSV at the end:

```powershell
uv run https://raw.githubusercontent.com/SainsburyWellcomeCentre/crabs-exploration/orientation-labelling-v1/scripts/orientation/annotate_crabs_orientation_manual.py <package> --annotator <name> --resume <labels folder>\orientation_lines_<name>_<timestamp>.csv
```

where `<labels folder>` is the path to your `manual_labels_<timestamp>`
folder. Your lines come back, and the same files are updated: no new ones
are made.

## 7. Send back your labels

When every line is green, close the window, and send us the whole
`manual_labels_<timestamp>` folder, CSV included. The CSV keeps the red
and yellow lines, which the results do not.

## For the maintainer

The labelling package is built from the `bboxes_and_masks_<timestamp>.zarr`
stores written by
[generate_masks_from_bboxes.py](../scripts/orientation/generate_masks_from_bboxes.py).
Use stores written on `main` after PR #298: older stores have empty root
attributes, and the selection script stops with an error on them.

### Build a labelling package

From the repo root, on ceph:

```bash
uv run scripts/orientation/create_labelling_package.py \
    /ceph/zoo/processed/CrabField/ramalhete_2023/CrabLabels/sep2023-full/annotations/bboxes_and_masks_<timestamp>.zarr \
    /ceph/zoo/processed/CrabField/ramalhete_2023/CrabLabels/aug2023-full/annotations/bboxes_and_masks_<timestamp>.zarr \
    --output-dir /ceph/zoo/processed/CrabField/ramalhete_2023/CrabLabels
```

By default, it samples 2 frames per `Right` video with seed 42
(`--frames-per-video`, `--camera`, `--seed`): 28 frames from 14 videos in
`sep2023-full`, and 10 frames from 5 videos in `aug2023-full`. It writes
`labelling_package_<timestamp>/`, with one subfolder per store holding
the selected frames and a slice of the store with the boxes only. The
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

The napari version, and its Qt backend (PyQt6: napari deprecates PyQt5),
are pinned in the script's `uv` header and in the `test_gui` CI job:
change both together.
