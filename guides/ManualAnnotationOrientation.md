# Manual labelling of crab orientation

This guide describes how we generate ground truth for the orientation of each crab.
We will manually draw a line over each annotated crab, following its long body axis.


## Requirements
* About 1 GB of disk space for the annotation data (can vary depending on the number of frames)
* The command line tool [`uv`](https://docs.astral.sh/uv/).

The steps below are for Windows, but the same commands work on macOS and Linux in a
terminal.

## Labelling procedure

Below we describe the steps to follow to label the ground truth data.

## 1. Install uv

We use `uv` to install Python and the relevant dependencies. To install it, open **PowerShell** and run:

```powershell
powershell -ExecutionPolicy ByPass -c "irm https://astral.sh/uv/install.ps1 | iex"
```

Then close PowerShell and open it again, so it finds `uv`. To check it
worked, run `uv --version`.

If the `uv` is too old, the tool will stop with a message asking you
to run `uv self update`. Run it, then try again.

See the [uv documentation](https://docs.astral.sh/uv/getting-started/installation/#__tabbed_1_2) for more details.

## 2. Copy the labelling package

Copy the labelling package folder from ceph to your computer.

The name of the labelling package is in the format `labelling_package_<timestamp>`
and is about 0.6 GB. It has two subfolders:

* `sep2023-full`
* `aug2023-full`

Each subfolder holds a
set of frames to label and the boxes manually drawn on them. Do not change anything inside
the labelling package.

Your orientation labels will be saved next to the labelling
package, in a new `manual_labels_<timestamp>` folder (with matching `<timestamp>`).

Make a note of the path to the labelling package in your computer for the next step.

> [!NOTE]
> To copy the path to a folder in Windows, go to File Explorer,
> Shift + right-click the folder and choose **Copy as path**.

## 3. Launch the napari labelling tool

In PowerShell, run the command below, with:

- the path to your copy of the labelling package, in place of `<package>`;
- your name in place of `<name>`, with letters, digits and `-` only (no
  spaces or `_`), for example `jane-doe`. **Use the same name every time.**

```powershell
uv run https://raw.githubusercontent.com/SainsburyWellcomeCentre/crabs-exploration/orientation-labelling-v1/scripts/orientation/annotate_crabs_orientation_manual.py \
<path/to/labelling/package> \
--annotator <name>
```

The command points to a fixed version of the tool (`orientation-labelling-v1`), so everybody labels with the same code.

The first run may take a few minutes, while `uv` downloads the dependencies that the tool needs. Later runs start in seconds.

Once all dependencies are installed, a [napari](https://napari.org/) window opens with three layers:

- **frames**: hold the frames of both datasets, one after the other. Move between frames with
  the keyboard side arrows or the slider under the image;
- **bboxes**: the annotated bounding boxes, in magenta, each with its `id` in the top-left corner. The bounding boxes
  cannot be edited;
- **orientation lines**: an empty layer, pre-selected and ready to draw lines in it;

The canvas additionally shows:
- your **progress**, in the top-left corner of the image, as how many boxes of
  this frame have a green line;
- the **features table**, on the right, listing the lines added to the frame you
  are on;
- the **show** panel, on the right, to hide lines and boxes.


## 4. Start labelling

For every box on every frame, draw one line **parallel to the long axis of the crab**.

To draw a line, click and drag over the crab, then release.

Before starting to draw lines, remember:
- Only the line's angle will be used: the length of the line is not important neither its position.
- A line is linked to the box that contains **its midpoint**. So we need to draw it over the box, with its midpoint roughly inside the box. A line and a box that are "linked" share the same `id`. By default, a linked box becomes hidden (see "Show panel" below).
- If you cannot tell a crab's orientation (because it is blurry, hidden or otherwise unclear), leave its box without a line. No label is better than a wrong one.


### User interface tips

You may find some of these help you label more comfortably:
- You can zoom in and out with the mouse wheel.
- You can move around the image while drawing (i.e., to pan) by holding the spacebar and dragging the image with your mouse.
- You can toggle the bounding boxes layer visibility by selecting the `bboxes` layer and pressing `V`. Remember to select the `orientation lines` layer again before you
  draw.
- To hide the `id` numbers of the lines or of the boxes, select the layer and untick **display text** in the layer controls, at the top left.

There are also some keyboard shortcuts that can be helpful. The following only work when the `orientation lines` layer
is selected in the layer list.

| Key | What it does |
|---|---|
| `L` | draw lines ("add line" mode) |
| `S` | select and move lines ("select" mode) |
| `Delete` | delete the selected line |


### Line colours

When you draw a line, it turns from white into one of three colours. Each colour reflects the line's linked status:

| Colour | Meaning | What to do |
|---|---|---|
| green | the line is linked to a single box (and shares the box's `id`) | nothing |
| yellow | Two options: (1) the midpoint is inside 2 or more boxes, so the `id` is ambiguous and set to `-1`; or (2) another line on the frame has the same `id` | type the right `id` in the features table, or move the line |
| red | the line's midpoint is not inside a box, or the `id` typed is a box that does not contain the midpoint | move the line into a box, or delete the line if spurious |

A warning message appears in the bottom-right corner when a new line is yellow or red with more details and suggestions.

**Only green lines are saved as results.** Before you finish labelling, check that every line added is green.

> [!NOTE]
> You can use the options in the bottom right "Show" panel to review your labels at the end. For example, you can select only the "remaining boxes" option to review which boxes across frames still don't have a line linked to them.

Boxes with a green line are called "labelled" in the top left progress panel. Boxes you leave without a line on purpose (e.g. because it is unclear) are not counted as labelled, so the count may not reach the total.

### How to manually set the id of a line in the features table

1. Make sure the `orientation lines` layer is selected in the layer list.
2. In the features table, find the line's row. The table only lists the
   lines of the frame you are on. The `Index` column is the line number
   given in the warning message.
3. Double-click its `id` cell, type the `id` of the right box, and press
   `Enter`.


### How to move or delete a line

- Press `S` (select mode), click the line and drag it. Press `L` to go back
  to drawing lines.
- **Moving a line guesses its `id` again** from its new position, and
  replaces any `id` you typed. Type it again if needed.
- To delete a line, select it and press `Delete`.


### Show panel

The **show** panel in the bottom right allows us to visualise or hide lines and boxes.

The **labelled boxes** checkbox starts unticked, so a box disappears once it has a green line, and you see only the boxes left to label. To always see the labelled boxes, tick this checkbox.

> [!WARNING]
> Lines that are hidden can still be selected and moved, so remember to tick them again before you edit lines near them.


## 5. Output files

Your work is saved automatically after every change in a
`manual_labels_<timestamp>` folder next to the labelling package. The `<timestamp>` in the file names is when you
started labelling, and it is shared with the labelling package's timestamp.

The `manual_labels_<timestamp>` folder holds:
- `orientation_lines_<name>_<timestamp>.csv`: your work in progress. It is
  saved after every change, and holds all your lines, red and yellow ones
  included. This is the file you need to resume a labelling session.
- `orientation_lines_<name>_<timestamp>.csv.bak`: a copy of the CSV from
  before your last resume, in case lines are deleted by mistake.
- `sep2023-full/…zarr` and `aug2023-full/…zarr`: the labelled green
  lines only. They are written when you close the napari window, and are rebuilt from the
  CSV each time. If one is missing or looks wrong, resume and close the window again.

Do not edit, rename or move these files.

```text
labelling_package_<timestamp>/        (the package: do not change it)
manual_labels_<timestamp>/
├── orientation_lines_<name>_<timestamp>.csv
├── orientation_lines_<name>_<timestamp>.csv.bak    (backup file, only after a resume)
├── sep2023-full/orientation_<name>_<timestamp>.zarr
└── aug2023-full/orientation_<name>_<timestamp>.zarr
```

Do not use napari's **File > Save** (`Ctrl + S`): there is no need. That menu exports layers in napari's own format, which we do not use.

## 6. Stopping and resuming a session

To stop a labelling session and save your current progress, close the napari window.

To continue labelling later, or after a crash, run the same command as in step 3, but add
`--resume` and the path to your CSV file at the end:

```powershell
uv run https://raw.githubusercontent.com/SainsburyWellcomeCentre/crabs-exploration/orientation-labelling-v1/scripts/orientation/annotate_crabs_orientation_manual.py \
<path/to/labelling/package> \
--annotator <name> \
--resume <manual_labels_<timestamp> folder>\orientation_lines_<name>_<timestamp>.csv
```

Then you can continue labelling and the data is saved to your existing files.

## 7. Send back your labels

When you are done labelling, check that every line added is green. Then close the napari window, and share the whole
`manual_labels_<timestamp>` folder, CSV included. The CSV keeps the red and yellow lines, which the results do not.
