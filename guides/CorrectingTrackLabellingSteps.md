# Guide to correcting track labels

Authors: Sofía Miñano and Nik Khadijah Nik Aznan

The csv files that contained the raw track labelling will typically be in a subdirectory under
<!-- TODO -->
```
/ceph/zoo/users/sminano/crabs_bboxes_labels
```

Below we outline the steps on how to correct the labelling of these csv.

## Steps

### 0. Mount the server from `ceph` with the frames to label

- Remember that to mount the server you need to be connected to the SWC network (or SWC VPN)
- In Mac, mounting the server can be done in the Finder app, pressing CMD+K and writing the following address:
  ```
  smb://ceph-gw02.hpc.swc.ucl.ac.uk
  ```
- Once the directory with the csv file is mounted, we can access it under `Volumes` in Mac. For example, its path could look like:
<!-- ToDO -->
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

  - `Project Name`: use the name of the csv file and add `_corrected_`+ your initials as a suffix. For example, for Sofia:
    ```
    05.09.2023-05-Left_corrected_SM
    ```
  - `Default Path`: this will be the starting point of the file explorer when loading the images to label, so we want the folder with the frames we want to label here. For the example above, the default path to input would be:
<!-- ToDO -->
  ```
  /Volumes/zoo/users/sminano/crabs_bboxes_labels/20230816_ramalhete2023_day2_combined/
  ```

> [!IMPORTANT]
>
> A trailing slash is needed!

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

### 4. Correcting the labels

- Now you see all the images and the bounding boxes with track id.
- Make sure the track id is shown instead of the image number. 
    - To do so, press up key, it will toggle between `region label cleared `or `region label set to region attribute [track]`. We want the second one `region label set to region attribute [track]`
    - Pressing bottom key will change the colour of the bounding box.
- If it is easier to go by track id, go from id 1 follow the id in between frames. You can just press the right key and see the movement of the bounding box and the track id.
- If you cannot easily find the number you want, click toggle annotation editor, go to the track you want, it will be highlighted in the frame. 
    - Be careful now, when it is highlighted, if you press right key, the bounding box will be shifted to the right. Just click anywhere to disable the highlight.
- If the id you are following is missing the bounding box, you can copy `ctrl+c` from the previous frame and then paste `ctrl+v` on the current frame, you can adjust the bounding box to where the crab is either by dragging with your mouse or use arrow keys.
- If the id you are following has been re-id (assign to a new id):
    - Go to the menu bar, choose the 6th icon after `help` (it looks like grid). 
    - All the frames will be group in a grid style. 
    - Choose `Group by` `region [track]` - the track number will be popped up next to it. 
    - Choose the track number the id has been re-id.
        - For example: the original id is 9, the crab is now been re-id to 78.
        - Choose `78/188: track = 78`
    - Click Toggle Annotation Editor at the side. 
        - if you has this shown, unclick and click again. 
        - make sure the track shows the track you choose only instead of `xxx different values:...`
    - Change the track in there, from the one is being re-id to the original one. 
        - for example from 78 to 9
    - Then click anywhere and click the same icon again so you will get only one frame back. 
    - See if the changes happened correctly.
    - You do not need to wait the grid images finished loading before doing any of the step above.
- If the crab has been re-id for one frame and then lost track again, maybe, change in that frame will be easier than the grid method. 
    - Just click the bounding box to get it highlighted. 
    - The box with track and track number will be appeared (make sure the toggle annotation editor is closed by clicking the toggle annotation editor). 
    - Simply key in the correct id there.

> [!TIP]
>
> Browser:
> - I tried using Edge and Safari. Safari is more temperamental, randomly reload the page and you will lose the unsaved work. Save regularly. So far Edge has been good.

> Zooming in:
>
> - It is possible to zoom in the frame to label using the magnifying glass tool in the top bar.
> - However, I found that when I am zoomed in on the right hand side of the frame, if I switch to the next frame (with the left arrow), the zoom location is reset to the leftmost-side of the image. This is a bit annoying when labelling the right hand side of the image.
> - As a workaround I did the following (using Chrome, in a Mac): instead of zooming in the frame, we zoom in the whole browser window. To do this, I click on an empty area of the left side panel and zoom-in using the pinch gesture on the trackpad. This way we can zoom in on the right hand side of the image and also switch across frames without the zoom changing location.
> - For some reason, this pinch gesture is not equivalent to clicking the magnifying glass tool in Chrome - I'm not sure why

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


### 6. Export the annotations

We save our work in two formats: as a VIA json file and as csv annotations.

To save a VIA json file:

- In the top bar, click `Project` > `Save Project`
- Leave all settings `ON` (as by default) and click `OK` to export
- This generates a json file with the name `<project_name>.json` that we can load again later in VIA to continue labelling.

To export the annotations to COCO format:

- in the top menu, click `Annotation` > `Export annotations (as csv)`
- This will download a json file to your Downloads folder named `<project_name>_csv.csv`

> [!TIP]
>
> - Save the data frequently, at least every frame - remember there is no undo button! 😬
> - It is simpler to save the data locally (to your laptop) to start with, rather than in `ceph`.

Once you are done with all your annotations for the frames you imported, please upload the 2 final json files (the VIA json file and COCO json file) to the subdirectory `annotations`, which is next to the extracted frames. In our previous example, that subdirectory would be:
<!-- TODO -->
```
/ceph/zoo/users/sminano/crabs_bboxes_labels/20230816_ramalhete2023_day2_combined/annotations
```

### 7. Reloading an unfinished project
<!-- TODO -->
