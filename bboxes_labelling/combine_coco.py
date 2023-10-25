import json
import argparse


def combine_coco(coco1: str, coco2: str) -> None:
    """
    Combine two COCO format JSON files and save the combined data to a new file.

    Parameters:
        coco1 (str): Path to the first COCO format JSON file.
        coco2 (str): Path to the second COCO format JSON file.

    Returns:
        None
    """
    # Load the contents of the first JSON file
    with open(coco1, "r") as file1:
        data1 = json.load(file1)

    # Load the contents of the second JSON file
    with open(coco2, "r") as file2:
        data2 = json.load(file2)

    # Calculate the offset for image and annotation IDs in the second dataset
    offset_image_id = max([image["id"] for image in data1["images"]])
    offset_annotation_id = max(
        [annotation["id"] for annotation in data1["annotations"]]
    )

    # Update the image and annotation IDs in the second dataset
    for image in data2["images"]:
        image["id"] += offset_image_id
    for annotation in data2["annotations"]:
        annotation["id"] += offset_annotation_id
        annotation["image_id"] += offset_image_id

    # Combine the images and annotations from both datasets
    combined_images = data1["images"] + data2["images"]
    combined_annotations = data1["annotations"] + data2["annotations"]

    # Create a new COCO dataset dictionary
    combined_data = {
        "images": combined_images,
        "annotations": combined_annotations,
        "categories": data1["categories"],
    }

    # Extract filenames without extensions from input paths
    filename1 = coco1.split(".")[0]
    filename2 = coco2.split(".")[0]

    # Create a new filename for the combined JSON file
    combined_filename = f"{filename1}_{filename2}_combine.json"

    # Save the combined data to the new JSON file
    with open(combined_filename, "w") as combined_file:
        json.dump(combined_data, combined_file)


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
        "--coco_one_path",
        type=str,
        required=True,
        help="Path for the the first coco file to be combined",
    )
    parser.add_argument(
        "--coco_two_path",
        type=str,
        required=True,
        help="Path for the the second coco file to be combined",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = argument_parser()

    combine_coco(args.coco_one_path, args.coco_two_path)
