# Steps

1. Install uv
    - Follow the installation instructions [here](https://docs.astral.sh/uv/getting-started/installation/#__tabbed_1_2)


2. Download the orientation labelling script from the crabs-exploration GitHub repository
    - Run `curl ..... https://raw.githubusercontent.com/SainsburyWellcomeCentre/crabs-exploration/refs/heads/main/scripts/burrows/annotate_crabs_orientation_manual.py` -- check this URL


3. Download annotation frames locally from ceph
    - From G-Node GIN, download zip (they need password), OR
    - Copy zarr stores with bboxes (& masks) from ceph to a local destination in your computer
    - Ceph location:
        - /ceph/zoo/processed/CrabField/ramalhete_2023/CrabLabels/aug2023-full/bboxes_and_masks_20261015_194318.zarr
        - /ceph/zoo/processed/CrabField/ramalhete_2023/CrabLabels/sep2023-full/bboxes_and_masks_20261015_200236.zarr


3. Run script via `uv run`
    ```
    uv run /path/to/downloaded/script
    ```
    It will launch an interactive napari viewer with the following layers:
    - an image layer with a subset of the frames for aug and sept
    - a shapes layer with the bboxes annotations per frame **showing their ID as text labels** (read from zarr)
    - a shapes layer with pre-selected shape Line, which the user will use to annotate orientation


4. Label orientation
    - For every bbox per frame, draw a line that is parallel to the "long axis" of the crab (medial-lateral axis)
        - To draw a line: hold-click and drag
        - We don't need for the centre of the line to be at the centre of the box, or for the line length to represent the width of the animal.
        - Each line should fall within a box. Lines outside all boxes will be discarded.
        - Ideally each line will intersect one box only. If there are overlapping boxes.... ?
        - If it is helps the labelling, once you locate the box you can toggle the visibility of the bboxes layer with one of these shortcuts:
            • Toggle selected layer visibility: V
            • Toggle unselected layers visibility: Shift + V
            • Show only the layer above (cycle/isolate): Shift + Alt + Up
            • Show only the layer below (cycle/isolate): Shift + Alt + Down


5. Saving the data
    - The output format follows the one for the input zarr data (same coordinates, but only with the theta variable?)
    - During labelling, we have Autosaving (like in https://github.com/SainsburyWellcomeCentre/crabs-exploration/blob/main/scripts/burrows/annotate_burrow_prompts_manual.py) to ensure no data is lost if the program crashes
    - We can also restart an unfinished session



# Questions / points to discuss
- output format
    - sth as close as possible to the input zarr, so that we can easily compute the error in theta
        - same coordinates, but only with the theta variable?
    - I think the input uses `time` axis (movement-like) rather than `image_id` axis (ethology-like). Should I change that?
        - See https://ethology.neuroinformatics.dev/latest/examples/coco_bbox_ethology_and_movement.html#transform-dataset-to-a-movement-like-dataset

- pre-computed subset of frames
    - we have 27 videos, but only a subset of them is present in the annotated data
		- In the Sept annotations, we have frames from 14 'right' videos and 14 'left' videos. We don't have annotations from 8 'right' videos and 8 'left' videos.
		- In the Aug annotations, we have frames from 5 'right' videos and 6 'left' videos. That covers the full set of August videos.
    - So we can ask users to label orientation from a subset of 'right' video frames in the annotation set
		- 'right' because those are the only ones we run inference on, so we can ignore 'left' videos for now. That is a pool of 14+5 = 19 videos.
		- We can select 2 annotated frames at random per video ---> 19*2 = 38 frames

    - Some other thoughts
		- can I use ethology (https://ethology.neuroinformatics.dev/latest/api_index.html) for this? would it be useful or overkill? (I prefer simplicity)
        - I think this will be a separate script, that we run before the labelling.
            - Then its output (a csv?) is passed as input to the labelling script
            - Is there a simpler way?
            - We could do the sampling of the frames in the labelling script and make it deterministic with a seed... is that better?
            - We may have to label more frames going forward. How can we make it simple but flexible? Could we maybe accept an input csv file with frames to exclude from the selection (because they have already been labelled)? I prefer simplicity above all for now, so if this becomes cumbersome we can make a note for the future on how to extend it and keep it very simple for now.
	- The selection of frames should be saved in the zarr output too
        - I think it is enough if this info is saved as a coordinate?

- bboxes napari layer
    - it is read from the input 'extended' zarr
        - (generated from the script at smg/gt-ellipses-prediction)
    - it shows the 'individual' per bbox as a `display text`
        - (this is so that the manual annotator can solve ambiguities re linking lines to bboxes)
        - `individual` would be a feature in the feature table, but this feature should not be editable
    - we should lock the bboxes layer so that the bbox annotations are not edited by mistake
        - `layer.editable = False`
        - but we should allow toggle visibility

- data saving while labelling
    - we should have an autosave feature
	- we should have a resume session feature
    - check how the saving was done for manual selection of burrow masks (see `scripts/burrows/annotate_burrow_prompts_manual.py`, `_autosave_napari_points()`)

- how to associated Lines (segments) to bboxes?
    - I use Line to refer to the segments the user draws because that is the name used by napari (but note these are not geometric lines but segments)
    - We can associate Lines to bboxes by overlap between line and box. So by placement: a line is associated with the box it mostly **intersects** with
    - We should add lines that are not correctly linked to a single bbox to a log file

- How to deal with lines that overlap with multiple boxes?
    - Since we dont care about the location of the line in the pixel space (meaning, we will only use the orientation aka theta of the line to compute the error), we can ask users to try to place the lines so that each line intersects only one box.
    - That may not be possible in all cases. So to deal with ambiguous cases:
        - We can use colours to highlight per frame the lines that are associated to a single box (green), ignored (red) or associated to multiple boxes (yellow).
        - Once added, lines can have a 'display text' label showing the 'individual' they are linked to.
        - We set the individual by guessing first: we guess the individual for each line based on the bbox they overlap most with
            - if there is only one box, the line turns green and shows the relevant `individual` label
            - if there are no boxes, `individual` is set to a placeholder value and the line turns red. A warning is shown to prompt the user to edit the individual label. If the individual is set to a value of a box the line does not intersect with, the line remains red.
            - if there are multiple boxes that intersect with the line, `individual` is set to a placeholder value and the line turns yellow. A warning is shown to prompt the user to edit the individual label. If the individual is set to a value of a box the line does not intersect with, the line turns red.
        - For 'red' or 'yellow' conflicts, maybe the simplest way is for users to edit the 'display text' associated to the line, which would be the `individual`
            - We can do this using the "Features table widget" plugin.
            - We should enable this plugin when launching napari (so that the user doesn't have to click around)
            - We could expose 'individual' as a feature in the table, and the user can use the plugin to review and edit their values
