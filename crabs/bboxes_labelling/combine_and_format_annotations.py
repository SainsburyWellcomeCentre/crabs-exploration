from pathlib import Path

from crabs.bboxes_labelling.annotations_utils import (
    combine_multiple_via_jsons,
    convert_via_json_to_coco,
)


def main(
    parent_dir_via_jsons,
    via_default_dir,
    via_project_name,
):
    # Get list of VIA JSON files
    list_json_files = [
        x
        for x in Path(parent_dir_via_jsons).glob("*")
        if x.is_file() and str(x).endswith(".json")
    ]
    list_json_files.sort()

    # Combine VIA JSONS
    json_out_fullpath = combine_multiple_via_jsons(
        list_json_files,
        via_default_dir=via_default_dir,
        via_project_name=via_project_name,
    )

    # Convert to COCO
    return convert_via_json_to_coco(json_out_fullpath)


if __name__ == "__main__":
    parent_dir_via_jsons = (
        "/Volumes/zoo/users/sminano/crabs_bboxes_labels/Aug2023/annotations"
    )
    via_default_dir = str(Path(parent_dir_via_jsons).parent)
    via_project_name = "Aug2023"

    coco_out_fullpath = main(
        parent_dir_via_jsons,
        via_default_dir,
        via_project_name,
    )
    print(coco_out_fullpath)
