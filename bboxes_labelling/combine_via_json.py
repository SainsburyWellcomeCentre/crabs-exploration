# %%
import json
from pathlib import Path
from typing import Optional


# %%
def combine_via_json(
    list_json_files: list,
    json_out_filename="VIA_JSON_combined.json",
    json_out_dir: Optional[str] = None,
) -> str:
    """Combine the input VIA JSON files into one

    Parameters
    ----------
    list_json_files : list
        list of paths to VIA JSON files

    Returns
    -------
    str
        _description_
    """

    via_data_combined = {}
    dict_of_via_img_metadata = {}
    list_of_via_img_id_list = []
    for k, js_path in enumerate(list_json_files):
        with open(js_path, "r") as js:
            via_data = json.load(js)

        if k == 0:
            via_data_combined["_via_settings"] = via_data["_via_settings"]
            via_data_combined["_via_attributes"] = via_data["_via_attributes"]
            via_data_combined["_via_data_format_version"] = via_data[
                "_via_data_format_version"
            ]

        # append dict of img metadata
        dict_of_via_img_metadata.update(via_data["_via_img_metadata"])
        # append list of img ids
        list_of_via_img_id_list.extend(via_data["_via_image_id_list"])

    via_data_combined["_via_img_metadata"] = dict_of_via_img_metadata
    via_data_combined["_via_image_id_list"] = list_of_via_img_id_list

    # Save the combined data to the new JSON file
    if not json_out_dir:
        json_out_dir = str(Path(list_json_files[0]).parent)

    json_out_fullpath = Path(json_out_dir) / json_out_filename

    with open(json_out_fullpath, "w") as combined_file:
        json.dump(via_data_combined, combined_file)

    return json_out_fullpath


# %%

if __name__ == "__main__":
    list_json_files = Path("/Users/sofia/Desktop/crabs_annotations_backup/NW").glob(
        "**/*"
    )

    list_VIA_files = [
        x
        for x in list_json_files
        if x.is_file() and str(x).endswith(".json") and "coco" not in str(x).lower()
    ]

    # combine_via_json(list_VIA_files)
