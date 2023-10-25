import json
import argparse
from typing import Dict, Any


def coco_conversion(json_file_path: str) -> None:
    """
    Convert annotation data in a JSON format to COCO format.

    Parameters:
        json_file (str): Path to the input JSON file containing annotation data.

    Returns:
        None

    This function takes annotation data in a specific JSON format and converts it
    into COCO format, which is widely used for object detection datasets.

    Note:
        The function assumes that the input JSON data has a specific structure
        that includes image metadata and regions of interest. The JSON data used
        has been produced by VIA.
    """

    # Load the JSON data
    with open(json_file_path, "r") as json_file:
        annotation_data = json.load(json_file)

    # Create COCO format data structures
    coco_data: Dict[str, Any] = {
        "info": {},
        "licenses": [],
        "categories": [{"id": 1, "name": "crab", "supercategory": "animal"}],
        "images": [],
        "annotations": [],
    }

    # Iterate through each image and annotation
    image_id = 1
    annotation_id = 1
    for image_filename, image_info in annotation_data["_via_img_metadata"].items():
        image_data: Dict[str, Any] = {
            "id": image_id,
            "width": 0,  # Set the image width here
            "height": 0,  # Set the image height here
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
                "category_id": 1,
                "bbox": [x, y, width, height],
                "area": width * height,
                "iscrowd": 0,
            }
            coco_data["annotations"].append(annotation_data)
            annotation_id += 1

        image_id += 1

    # Write the COCO data to a JSON file
    new_file_name = f"{json_file_path.split('.')[0]}_coco.json"
    with open(new_file_name, "w") as f:
        json.dump(coco_data, f)


def argument_parser() -> argparse.Namespace:
    """
    Parse command-line arguments for the script.

    Returns
    -------
    argparse.Namespace
        An object containing the parsed command-line arguments.
        The attributes of this object correspond to the defined
        command-line arguments in the script.
    """
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--json_path",
        type=str,
        required=True,
        help="Path for the json saved from VIA",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = argument_parser()

    coco_conversion(args.json_path)
