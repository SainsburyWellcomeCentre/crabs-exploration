"""Utility functions to work with annotations in JSON format."""

import json
import os
import re
from pathlib import Path
from typing import Any


def read_json_file(
    file_path: str,
) -> dict:
    """Read JSON file as dict.

    Parameters
    ----------
    file_path : str
        Path to the JSON file

    Returns
    -------
    Optional[dict]
        Dictionary with the JSON data

    """
    try:
        with open(file_path) as file:
            return json.load(file)
    except FileNotFoundError as not_found_error:
        msg = f"File not found: {file_path}"
        raise ValueError(msg) from not_found_error

    except json.JSONDecodeError as decode_error:
        msg = f"Error decoding JSON data from file: {file_path}"
        raise ValueError(msg) from decode_error


def combine_multiple_via_jsons(
    list_input_json_files: list,
    exclude_pattern: str | None = None,
    json_out_filename: str = "VIA_JSON_combined.json",
    json_out_dir: str | None = None,
    via_default_dir: str | None = None,
    via_project_name: str | None = None,
) -> str:
    r"""Combine all the input VIA JSON files into one.

    A VIA JSON file is a json file specific to the VIA tool
    that defines the annotations and also the visualisation settings
    for the tool.

    Some attributes of the combined VIA JSON file are taken from
    the first input VIA JSON file:
    - _via_settings,
    - _via_attributes, and
    - _via_data_format_version

    Parameters
    ----------
    list_input_json_files : list
        list of paths to VIA JSON files
    exclude_pattern : Optional[str], optional
        a regex pattern to exclude specific files from the input list.
        By default, None. E.g.: "\w+_coco_gen.json$"
    json_out_filename : str, optional
        name of the combined VIA JSON file, by default "VIA_JSON_combined.json"
    json_out_dir : Optional[str], optional
        parent directory to the combined VIA JSON file. If None, the parent
        directory of the first VIA JSON file in list_json_files is used.
    via_default_dir : Optional[str], optional
        The default directory in which to look for images for the VIA project.
        If None, the value specified in the first VIA JSON file is used.
        If a path is provided it needs to be a full path.
    via_project_name : Optional[str], optional
        The name of the VIA project.
        If None, the value specified in the first VIA JSON file is used.

    Returns
    -------
    json_out_fullpath: str
        full path to the combined VIA JSON file

    """
    # Initialise data structures for the combined VIA JSON file
    via_data_combined = {}
    dict_of_via_img_metadata = {}
    list_of_via_img_id_list = []

    # Apply exclude pattern if required
    if exclude_pattern:
        list_input_json_files = [
            js
            for js in list_input_json_files
            if not re.search(exclude_pattern, str(js))
        ]
    list_input_json_files.sort()

    # loop through the input VIA JSON files
    for k, js_path in enumerate(list_input_json_files):
        # open VIA JSON file
        via_data = read_json_file(js_path)

        # take some attributes from the first VIA JSON file
        if k == 0:
            via_data_combined["_via_settings"] = via_data["_via_settings"]
            via_data_combined["_via_attributes"] = via_data["_via_attributes"]
            via_data_combined["_via_data_format_version"] = via_data[
                "_via_data_format_version"
            ]

        # append dictionary of the images metadata (contains annotations)
        dict_of_via_img_metadata.update(via_data["_via_img_metadata"])

        # append list of images' IDs
        list_of_via_img_id_list.extend(via_data["_via_image_id_list"])

    # add data to combined VIA dictionary
    via_data_combined["_via_img_metadata"] = dict_of_via_img_metadata
    via_data_combined["_via_image_id_list"] = list_of_via_img_id_list

    # If required: change _via_settings > core > default_filepath
    if via_default_dir:
        # check if trailing slash is present in the directory path
        # and add it if not
        if not via_default_dir.endswith(os.sep):
            via_default_dir = f"{via_default_dir}{os.sep}"
        # check path is a full path
        if Path(via_default_dir) != Path(via_default_dir).resolve():
            msg = "Default VIA directory is not a fullpath"
            raise ValueError(msg)

        # assign directory path to the VIA combined dictionary
        via_data_combined["_via_settings"]["core"]["default_filepath"] = (
            via_default_dir
        )

    # If required: change _via_settings > project > name
    if via_project_name:
        via_data_combined["_via_settings"]["project"]["name"] = (
            via_project_name
        )

    # Save the VIA combined data as a new JSON file
    # if no output directory is passed, use the parent directory
    # of the first VIA JSON file in the list
    if not json_out_dir:
        json_out_dir = str(Path(list_input_json_files[0]).parent)
    json_out_fullpath = Path(json_out_dir) / json_out_filename

    with open(json_out_fullpath, "w") as combined_file:
        json.dump(via_data_combined, combined_file)

    return str(json_out_fullpath)


DEFAULT_CRAB_CATEGORY = {"id": 1, "name": "crab", "supercategory": "animal"}


def convert_via_json_to_coco(
    json_file_path: str,
    coco_category: dict = DEFAULT_CRAB_CATEGORY,
    coco_out_filename: str | None = None,
    coco_out_dir: str | None = None,
) -> str:
    """Convert annotation data for one category from VIA-JSON format to COCO.

    This function takes annotation data in a VIA JSON format and converts it
    into COCO format, which is widely used for object detection datasets.
    It assumes that:
    - the input JSON data has a specific structure (VIA-format) that includes
    image metadata and regions of interest,
    - that all annotations are of the same COCO category, and
    - that 'iscrowd' = 0 for all annotations.

    Parameters
    ----------
    json_file_path : str
        Path to the VIA-JSON file containing the annotation data.
    coco_category : dict, optional
        Dictionary with the category ID, name and supercategory for all the
        annotations.
    coco_out_filename : str, optional
        Name of the COCO output file. If None (default), the input VIA JSON
        filename is used with the suffix '_coco_gen'
    coco_out_dir : str, optional
        Name of the output directory where to store the COCO file.
        If None (default), the file is saved at the same location as the
        input VIA JSON file.

    Returns
    -------
    str
        path to the COCO json file.

    """
    # Load the annotation data in VIA JSON format
    with open(json_file_path) as json_file:
        annotation_data = json.load(json_file)

    # Create data structure for COCO
    coco_data: dict[str, Any] = {
        "info": {},
        "licenses": [],
        "categories": [coco_category],
        "images": [],
        "annotations": [],
    }

    # Iterate through each image and annotation from VIA JSON
    image_id = 1
    annotation_id = 1
    for image_info in annotation_data["_via_img_metadata"].values():
        image_data = {
            "id": image_id,
            "width": 0,  # TODO: find how we can get this data from json
            "height": 0,  # TODO: find how we can get this data from json
            "file_name": image_info["filename"],
        }
        coco_data["images"].append(image_data)

        for region in image_info["regions"]:
            x, y, width, height = (
                region["shape_attributes"]["x"],
                region["shape_attributes"]["y"],
                region["shape_attributes"]["width"],
                region["shape_attributes"]["height"],
            )

            annotation_data = {
                "id": annotation_id,
                "image_id": image_id,
                "category_id": coco_category["id"],
                "bbox": [x, y, width, height],
                "area": width * height,
                "iscrowd": 0,
            }
            coco_data["annotations"].append(annotation_data)
            annotation_id += 1

        # update image_id
        image_id += 1

    # Export the annotation data in COCO format to a JSON file
    # if no filename provided: use VIA JSON filename + "_coco_gen.json"
    if not coco_out_filename:
        coco_out_filename = Path(json_file_path).stem + "_coco_gen.json"
    # if no output directory provided: use VIA JSON parent directory
    if not coco_out_dir:
        coco_out_dir = str(Path(json_file_path).parent)
    coco_out_fullpath = Path(coco_out_dir) / coco_out_filename

    with open(coco_out_fullpath, "w") as f:
        json.dump(coco_data, f)

    return str(coco_out_fullpath)
