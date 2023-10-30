import json
import os
from pathlib import Path
from typing import Optional, Any


def read_json_file(file_path):
    """_summary_.

    Parameters
    ----------
    file_path : _type_
        _description_

    Returns
    -------
    _type_
        _description_
    """
    try:
        with open(file_path) as file:
            return json.load(file)
    except FileNotFoundError:
        print(f"File not found: {file_path}")
        return None
    except json.JSONDecodeError:
        print(f"Error decoding JSON data from file: {file_path}")
        return None


def combine_all_via_jsons(
    list_json_files: list,
    json_out_filename: str = "VIA_JSON_combined.json",
    json_out_dir: Optional[str] = None,
    via_default_dir: Optional[str] = None,
    via_project_name: Optional[str] = None,
) -> str:
    """Combine the input VIA JSON files into one.

    A VIA JSON file is a json file specific to the VIA tool
    that defines the annotations and visualisation settings
    for the tool.

    Parameters
    ----------
    list_json_files : list
        list of paths to VIA JSON files
    json_out_filename : str, optional
        name of the combined VIA JSON file, by default "VIA_JSON_combined.json"
    json_out_dir : Optional[str], optional
        parent directory to the combined VIA JSON file. If None, the parent
        directory of the first VIA JSON file in list_json_files is used.
    via_default_dir : Optional[str], optional
        The default directory in which to look for images in the VIA project.
        If None, the value specified in the first VIA JSON file is used.
        A full path is required.
    via_project_name : Optional[str], optional
        The name of the VIA project.
        If None, the value specified in the first VIA JSON file is used.

    Returns
    -------
    json_out_fullpath: str
        full path to the combined VIA JSON file
    """
    # initialise data structures for the combined VIA JSON file
    via_data_combined = {}
    dict_of_via_img_metadata = {}
    list_of_via_img_id_list = []

    # loop thru input VIA JSON files
    for k, js_path in enumerate(list_json_files):
        # open VIA JSON file
        via_data = read_json_file(js_path)
        # with open(js_path, "r") as js:
        #     via_data = json.load(js)

        # take some attributes from the first element of the list
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

    # add data to VIA dictionary
    via_data_combined["_via_img_metadata"] = dict_of_via_img_metadata
    via_data_combined["_via_image_id_list"] = list_of_via_img_id_list

    # Optionally: change _via_settings > core > default_filepath
    if via_default_dir:
        # check if trailing slash is present and add it if not
        if not via_default_dir.endswith(os.sep):
            via_default_dir = f"{via_default_dir}{os.sep}"
        via_data_combined["_via_settings"]["core"]["default_filepath"] = via_default_dir

    # Optionally: change _via_settings > project > name
    if via_project_name:
        via_data_combined["_via_settings"]["project"]["name"] = via_project_name

    # Save the VIA combined data as a new JSON file
    if not json_out_dir:
        json_out_dir = str(Path(list_json_files[0]).parent)
    json_out_fullpath = Path(json_out_dir) / json_out_filename

    with open(json_out_fullpath, "w") as combined_file:
        json.dump(via_data_combined, combined_file)

    return str(json_out_fullpath)


def coco_conversion(
    json_file_path: str,
    coco_category_ID: int = 1,
    coco_category_name: str = "crab",
    coco_supercategory_name: str = "animal",
    coco_out_filename: Optional[str] = None,
    coco_out_dir: Optional[str] = None,
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
    coco_categories : list, optional
        List of dictionaries with the COCO categories
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
        path to the COCO json file. By default, the file
    """
    # Load the annotation data in VIA JSON format
    with open(json_file_path, "r") as json_file:
        annotation_data = json.load(json_file)

    # Create COCO format data structures
    # we assume all annotations are of the same category
    coco_categories = [
        {
            "id": coco_category_ID,
            "name": coco_category_name,
            "supercategory": coco_supercategory_name,
        }
    ]
    coco_data: dict[str, Any] = {
        "info": {},
        "licenses": [],
        "categories": coco_categories,
        "images": [],
        "annotations": [],
    }

    # Iterate through each image and annotation from VIA JSON
    image_id = 1
    annotation_id = 1
    for image_info in annotation_data["_via_img_metadata"].values():
        image_data = {
            "id": image_id,
            "width": 0,  # Set the image width here --- can we get from JSON?
            "height": 0,  # Set the image height here --- can we get from JSON?
            "file_name": image_info["filename"],  # image_filename #
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
                "category_id": coco_category_ID,
                "bbox": [x, y, width, height],
                "area": width * height,
                "iscrowd": 0,
            }
            coco_data["annotations"].append(annotation_data)
            annotation_id += 1

        # update image_id
        image_id += 1

    # Export the annotation data in COCO format to a JSON file
    if not coco_out_filename:
        coco_out_filename = Path(json_file_path).stem + "_coco_gen.json"
    if not coco_out_dir:
        coco_out_dir = str(Path(json_file_path).parent)
    coco_out_fullpath = Path(coco_out_dir) / coco_out_filename

    with open(coco_out_fullpath, "w") as f:
        json.dump(coco_data, f)

    return str(coco_out_fullpath)
