# Steps to follow for manual labelling

Steps derived from meeting Sofia & Nik on 7/7/23

0. Mount the server from `ceph` with the frames to label
    - Typically, this will be located somewhere in SWC's `ceph` shared file system (we probably want it to be at a backed up location)
    - Remember that to mount the server you need to be connected to the SWC network (or SWC VPN)
    - In Mac, mounting the server can be done in the Finder app, pressing CMD+K and writing the following address:
        ```
        smb://ceph-gw02.hpc.swc.ucl.ac.uk
        ```
    - The frames to label can be extracted from the selected videos using SLEAP's clustering algorithm, with the script at `bboxes labelling/extract_frames_to_label_w_sleap.py`. 
    - The frames extraction script can be run in the cluster using the bash script at `bboxes labelling/run_frame_extraction.sh`

1. Download and launch the annotation tool
    - Access the online version of the tool [here](https://www.robots.ox.ac.uk/~vgg/software/via/via.html) or download it locally [via this link](https://www.robots.ox.ac.uk/~vgg/software/via/) (go to "Downloads", and then the "Image annotator" section to download the zip file).
    - It is very lightweight and also can be run offline.
    - If the tool is downloaded locally, launch it by selecting the `via.html` file from the expanded zip archive


2. Define project settings

    If we are launching the tool for the first time for a new project:
    - Click on the gear icon on the top bar to go to "Settings"
    - Edit the following settings:
        - Add a "Project Name" for the project.
        - Add a "Default Path". This will be the starting point of the file explorer when loading images to label. Note that a trailing slash is needed!
            - For example: `/Volumes/scratch/sminano/crabs_bbox_labels/20230616_135052/`
        - We can leave the rest of the settings with their default values
        - Click "Save" 
        - NOTE: We found the default path bit doesn't work very reliably...

3. Load the images to label
    - On the left hand side, find the "Project" section and click on the "Add Files" button
        - The file explorer should start at the default path, but it doesn't do that consistently... :(
    - Select the images to load
        - Multiple selection is allowed, and also across folders at the same level (at least in Mac)
        - For now we recommend to load the frames video by video. 
        - This is because flicking through the frames of a video recorded with a static camera helps to identify the individuals. But right now the video name is not part of the image name, so loading frames from different videos will mix them all together (and may lead to 'clashes' if the same frame number is extracted in two videos?).
    - Once loaded, use the left/right arrows to navigate through the images

4. Add bounding boxes annotations to the images
    - Ensure the shape selected is "Rectangular" (see left panel, "Region shape" section)
    - For the first frame, draw a rectangle around every indidual
        - The number shown next to the annotation is the identifier for that bounding box. For now, we don't aim to reuse that as an identifier for the same individual across frames, but it could be useful in the future.
        - Aim to draw the box as tight as possible around the animal. This is just so that there is not much variation across different labellers.
        - Flicking through the frames may be useful to identify individuals, since the cameras are static.
        - To delete a bounding box, select it and click `d`
    - For the following frames, the next shortcuts may speed up the process. To label frame `f+1` based on the annotations in `f`:
        - Select all the bounding boxes in frame `f` by pressing `a`
        - Copy all the bounding boxes by pressing `c`
        - Move to the next frame `f+1` with the right arrow
        - Paste the bounding boxes in `f+1` by pressing `v`
        - Adjust their location by clicking and then dragging each box's centre
        - Adjust their size by clicking and then dragging one of their corners
    - You can also copy the bounding boxes to several previous or next frames (click the "Paste-n icon" on the top bar and read the instructions)

5. Add a dropdown attribute to all annotations

    To correctly export the annotations in the COCO dataset format we need to define a Region attribute of type "dropdown" or "radio". If we don't, the annotations section of the output json shows up as empty. 
    - I think the difference between "radio" and "dropdown" is only at the UI level. It just defines how the different options are displayed to the annotator.

    To define a "Region Attribute":
    - On the left panel, under "Attributes", select "Region attributes"
    - In the text field, write the name of the supercategory ("supercategory" is the name used by the COCO format).     
        - For example, we can use "animal" to be consistent with the original COCO dataset.
        - More info on categories and supercategories [here](https://www.immersivelimit.com/tutorials/create-coco-annotations-from-scratch)
        - More on available categories and supercategories [here](https://tech.amikelive.com/node-718/what-object-categories-labels-are-in-coco-dataset/)
    - Click on the "+" to add it to the list of region attributes. 
        - The full list is shown in the dropdown immediately below the input text field.
    - In the fields that appear after clicking "+", go to "Type" and select "dropdown".
        - If you now select a bounding box in the image, a dropdown menu to define its category shows up.
    - In the table below, add the categories that are part of the selected supercategory. 
        - For example, for our case we can add the category "crab".
        - Question: do we need a "background" category? I think we don't, because apparently the category_id=0 is reserved for the class empty (more on that later).
    - Each annotation will be linked to one of these categories
        - To select a category as a default for all the annotations, click on the radio button under the last column of the table. 
    
    To visualise the Annotator Editor, press spacebar
    

6. Exporting in COCO format

    To export the annotations to COCO format:
    - in the top menu, click "Annotation" > "Export annotations (COCO format)" 
    - this will download a json file named "<project_name>_COCO.json"

    To understand the fields in the exported json, please check [this](https://www.immersivelimit.com/tutorials/create-coco-annotations-from-scratch) nice description of how the COCO format is defined for object detection. Below a brief overview.

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
    - segmentation: a list of polygon vertices around the object. If only bboxes are defined, it seems from [here](https://www.section.io/engineering-education/understanding-coco-dataset/#segmentation) that these are defined as:
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



7. Saving and reloading an unfinished project

    To save a VIA project:
    - In the top bar, click "Project" > "Save Project"
    - Leave all settings ON (as by default) and click "OK" to export
    - This generates a json file with the name "<project_name>.json" that we can later load again in VIA to continue labelling.
    
    To load an existing VIA project:
    - Launch the VIA application
    - Click "Project" > "Load" and select the VIA project json file
    - We found that the images are not correctly located when loading an old file, even if the "Default Path" for the project is correctly set. To locate the images, click on the highlighted link to the "browser's file select" and re-load the files :(

## Formats 
We would like to produce a labelling that can fullfil the requirements of the following formats
- [COCO dataset format](https://cocodataset.org/#format-data)
- [COCO CameraTrap format](https://github.com/microsoft/CameraTraps/blob/main/data_management/README.md)
    - This would be for context-RCNN, it needs slightly further customisation though (see 3rd paragraph [here](https://github.com/tensorflow/models/blob/master/research/object_detection/g3doc/context_rcnn.md#generating-tfrecords-from-a-set-of-images-and-a-coco-cameratraps-style-json))

## Some thoughts
- We need to review how we deal with an image with no annotations (see 3rd paragraph [here](https://github.com/tensorflow/models/blob/master/research/object_detection/g3doc/context_rcnn.md#generating-tfrecords-from-a-set-of-images-and-a-coco-cameratraps-style-json); do we need an empty category for images? maybe we don't need anything special)
- We'll likely need to write a Python script to add custom fields to our COCO json exported file, as VIA doesn't seem well-suited for adding custom fields to it. This may be the case for the location field required in context-RCNN

- Is there a way to go through the bounding boxes with a keyboard shortcut?
- The VIA example was useful to troubleshoot a bit. We didn't check the manual in detail though.
- We may need to think about how we update/combine the files we generate while labelling each of us separately.

- I need to modify the frame extraction script so that the frames' name also include the video they belong to. The idea is that we later incorporate this info as a location_ID field in the json file
- Nik has experience with [LabelMe](https://github.com/CSAILVision/LabelMeAnnotationTool) as an annotation tool. She says it is similar to VIA so for now we stick to our initial one, but maybe it comes in handy in the future. It seems to have nice functionalities to label collaboratively, but it does seem a bit faffy to set up.