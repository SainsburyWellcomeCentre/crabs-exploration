# Guide to manual labelling

Authors: Sofía Miñano and Nik Khadijah Nik Aznan

The frames extracted for labelling will typically be in a subdirectory under

```
/ceph/zoo/users/sminano/crabs_bboxes_labels
```

Below we outline the steps on how to add annotations to these extracted frames.

## Steps

### 0. Mount the server from `ceph` with the frames to label

- Remember that to mount the server you need to be connected to the SWC network (or SWC VPN)
- In Mac, mounting the server can be done in the Finder app, pressing CMD+K and writing the following address:
  ```
  smb://ceph-gw02.hpc.swc.ucl.ac.uk
  ```
- Once the directory with the extracted frames is mounted, we can access it under `Volumes` in Mac. For example, its path could look like:
  ```
  /Volumes/zoo/users/sminano/crabs_bboxes_labels/20230816_ramalhete2023_day2_combined/
  ```

### 1. Download and launch the VIA annotation tool

- Access the online version of the tool [here](https://www.robots.ox.ac.uk/~vgg/software/via/via.html) or download it locally [via this link](https://www.robots.ox.ac.uk/~vgg/software/via/) (click on `Downloads` > `Image annotator` section and download the zip file).
- It is very lightweight, browser-based and can be run offline.
- If you download the tool locally, launch it by selecting the `via.html` file from the expanded zip archive

### 2. Define the project settings

If we are launching the tool for the first time for a new project:

- Click on the gear icon on the top bar to go to `Settings`
- Edit the following settings:

  - `Project Name`: use the name of the directory holding the extracted frames and add `_`+ your initials as a suffix. For example, for Sofia:
    ```
    20230817_ramalhete2023_day2_combined_SM
    ```
  - `Default Path`: this will be the starting point of the file explorer when loading the images to label, so we want the folder with the frames we want to label here.

    > [!IMPORTANT]
    >
    > A trailing slash is needed!

    For the example above, the default path to input would be:

    ```
    /Volumes/zoo/users/sminano/crabs_bboxes_labels/20230816_ramalhete2023_day2_combined/
    ```

  - Leave the rest of the settings with their default values
  - Click `Save`
  - NOTE: We found the default path bit doesn't work very reliably, so sometimes you will need to browse the image to label and select them manually.

### 3. Load the images to label

- If the images aren't loaded automatically:

  - On the left hand side, find the `Project` section and click on the `Add Files` button
  - Select the images to load. Multiple selection is allowed, and also across folders at the same level (at least in Mac 🍎)

> [!TIP]
>
> - Loading the frames of just one video in a labelling session is a good idea. This is because flicking through the frames of a video recorded with a static camera helps us identify the individuals when they move.

- Once loaded, use the left/right arrows to navigate through the images

### 4. Add bounding boxes annotations to the images

- Before starting, ensure the shape selected is `Rectangular` (see left-hand side panel, `Region shape` section)

- Zoom-in the frame to label it

  > [!TIP]
  >
  > Zooming in:
  >
  > - It is possible to zoom in the frame to label using the magnifying glass tool in the top bar.
  > - However, I found that when I am zoomed in on the right hand side of the frame, if I switch to the next frame (with the left arrow), the zoom location is reset to the leftmost-side of the image. This is a bit annoying when labelling the right hand side of the image.
  > - As a workaround I did the following (using Chrome, in a Mac): instead of zooming in the frame, we zoom in the whole browser window. To do this, I click on an empty area of the left side panel and zoom-in using the pinch gesture on the trackpad. This way we can zoom in on the right hand side of the image and also switch across frames without the zoom changing location.
  > - For some reason, this pinch gesture is not equivalent to clicking the magnifying glass tool in Chrome - I'm not sure why

- To label the **first frame**, draw a rectangle around every individual

  - To select a bounding box, click on it.
  - To unselect a selected bounding box, press `ESC`.
  - To delete a bounding box, select it and press `d`.
  - To resize a bounding box, click and drag its borders.
  - Some additional comments
    - The number shown next to the annotation is the identifier for that bounding box. For now, we don't aim to reuse that as an identifier for an individual, so don't worry about being consistent with that ID across frames.
    - Aim to draw a fairly tight box around the animal, but including all of its bodyparts. This is just so that there is not much variation across different labellers.
    - The bounding boxes can overlap - that is not an issue.
    - Aim to draw a bounding box also if the animal is partially occluded.
    - Flicking through the frames before and after the current one is very useful to identify individuals.
    - The `Help` menu in the top bar contains further documentation on the annotation tool and its shortcuts.

- For the next frames, the following shortcuts can speed up the process. To label frame `f+1` based on the annotations in `f`:

  - Select all the bounding boxes in frame `f` by pressing `a`
  - Copy all the bounding boxes by pressing `c`
  - Move to the next frame to label `f+1` with the right arrow
  - Paste the bounding boxes in `f+1` by pressing `v`
  - Adjust their location by clicking and then dragging each box's centre
  - Adjust their size by clicking and then dragging one of their corners
  - Delete the bounding boxes that don't apply anymore with `d`

- You can also copy the bounding boxes to several previous or next frames (click the `Paste-n icon` on the top bar and read the instructions)

  - I found this less useful

- Some convenient shortcuts:

  - Press `b` to toggle the boxes visibility
  - Press `l` to toggle the visibility of the boxes' IDs
  - Press `spacebar` to show the annotations for that frame

> [!CAUTION]
> There is no undo action in the annotation tool!

- Be sure to save frequently, at least every labelled frame
  - more on saving the annotations in steps 6 and 7.
- If you are using the "copy+paste" approach to reuse the annotations of the previous frame: note that the copied annotations stay in the clipboard after pasting!

  - Be careful that you don't press `v` accidentally, as that would paste all boxes again to the frame you are viewing (and deleting them one by one will be painful)

- Try to be as consistent as possible with other labellers (in terms of what you are labelling as a crab and how)

### 5. Add a dropdown attribute to all annotations

- This is an important step! But we need to do it only once
- To correctly export the annotations we need to define a `Region attribute` of type `dropdown`.
- In our case the same region attribute will be applied to all labels (since all are crabs)
- We can do this at any stage during labelling, but make sure you do it before you finish defining all labels.

#### To define a "Region Attribute":

- On the left panel, under `Attributes`, select the `Region attributes` tab.
- In the text field for `attribute name`, write the name of the supercategory
  - We use `animal` to be consistent with the original COCO dataset.
- Click on the `+` symbol to add `animal` to the list of region attributes.
  - The full list of defined supercategories is shown in the dropdown immediately below the input text field -- in our case it's only one.
- From the fields that appear after clicking `+`, click on `Type` and select `dropdown`.
  - If you now select a bounding box in the image, a dropdown menu to define its category shows up.
- In the table below,:
  - add `crab` under `id`
  - add `crab` under `description`
  - select the radio button under `def` - this sets all annotations to be `crabs` by default. If we don't do these none of the annotations will be labelled.
- To visualise the Annotator Editor for the current frame, press `spacebar`
  - The dropdown menu for a bounding box does not show up if annotator editor is visible.

### 6. Export the annotations

We save our work in two formats: as a VIA json file and as COCO annotations.

To save a VIA json file:

- In the top bar, click `Project` > `Save Project`
- Leave all settings `ON` (as by default) and click `OK` to export
- This generates a json file with the name `<project_name>.json` that we can load again later in VIA to continue labelling.

To export the annotations to COCO format:

- in the top menu, click `Annotation` > `Export annotations (COCO format)`
- This will download a json file to your Downloads folder named `<project_name>_COCO.json`

> [!TIP]
>
> - Save the data frequently, at least every frame - remember there is no undo button! 😬
> - It is simpler to save the data locally (to your laptop) to start with, rather than in `ceph`.

Once you are done with all your annotations for the frames you imported, please upload the 2 final json files (the VIA json file and COCO json file) to the subdirectory `annotations`, which is next to the extracted frames. In our previous example, that subdirectory would be:

```
/ceph/zoo/users/sminano/crabs_bboxes_labels/20230816_ramalhete2023_day2_combined/annotations
```

### 7. Reloading an unfinished project

To load an existing VIA project:

- Launch the VIA application
- Click `Project` > "Load" and select the VIA json file (`<project_name>.json`)

Troubleshooting reloading

- We found that sometimes the images are not correctly located when loading an old file, even if the `Default Path` for the project is correctly set in the `Settings`. To locate the images, click on the highlighted link to the `browser's file select` and re-load the files manually (multiple selection is allowed).
- After re-loading a project, the project name shown in the left hand side may be different to the one we assigned originally. However, we found that even if that field is not modified, when we save the project as a json file the original name is used.
- If you find the bounding boxes you are intending to draw are weirdly offset or if you see other odd behaviour, simply save the VIA project and start fresh in a new browser session.

---

## Appendix: COCO format

To understand the fields in the exported COCO json file, please check [this](https://www.immersivelimit.com/tutorials/create-coco-annotations-from-scratch) nice description of how the COCO format is defined for object detection. Below a brief overview.

Under **images**, we have a list of the images in the project. The relevant fields in each image are:

- id,
- filename. We may want to include the video the frame was extracted from in the filename so that it is reflected here.
- The [COCO CameraTrap format](https://github.com/microsoft/CameraTraps/blob/main/data_management/README.md) adds a "location" field to each image (which is string by convention) - we may need to add this if we plan to use the context-RCNN model.

Under **annotations**, we have a list of the annotations we have defined. In our case all annotations are bounding boxes. Each bounding box will have these fields:

- id: id of this specific annotation,
- image_id: id of the image it is defined on,
- bbox: the format is [top left x position, top left y position, width, height], in absolute, floating-point coordinates,
- area: width x height of the bounding box,
- category_id: id of its category (specified in the categories section),
- segmentation: a list of polygon vertices around the object. If only bounding bboxes are defined, it seems from [here](https://www.section.io/engineering-education/understanding-coco-dataset/#segmentation) that these are defined as:
  ```
  [xmin, ymin, xmin, ymin + ymax, xmin + xmax, ymin + ymax, xmin + xmax, ymax].
  ```
  That is, it is a list of coordinates in the following order: top-left corner, top-right corner, bottom-right corner and bottom-left corner
- is_crowd: whether the segmentation is for a single object (0) or for a group/cluster of objects (1). In principle we don't plan to use this.

Under **categories**, we have a list of the categories we have defined. Each one will have:

- id: each category id must be unique (among the rest of the categories). Non-negative integers only.
  - According to the [COCO CameraTrap format](https://github.com/microsoft/CameraTraps/blob/main/data_management/README.md), the category_id = 0 is reserved for the class "empty" (is this true for the general COCO dataset format too?)
- supercategory: the supercategory this category belongs to
- name: a descriptive name for this category.
