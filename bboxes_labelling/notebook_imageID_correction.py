# ImageID correction pipeline

# %%
from pathlib import Path
from convert_coco import coco_conversion
from combine_coco import combine_coco


# %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
# Path to VIA project JSON files
parent_dir_via_json_files = Path("/Users/sofia/Desktop/crabs_annotations_backup/NW")

list_patterns_to_exclude = ["_coco.json", "_coco_regen", "combined"]

# %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
# Convert VIA project JSON files to COCO JSON files (regenerated)

# get list of json files
list_json_files = [
    x
    for x in parent_dir_via_json_files.glob("**/*")
    if x.is_file() and str(x).endswith(".json")
]
list_json_files.sort()

# exclude those whose filenames match the patters
list_VIA_JSON_files = list_json_files.copy()
for f in list_json_files:
    if any([x in str(f) for x in list_patterns_to_exclude]):
        list_VIA_JSON_files.remove(f)


# build a COCO file for each VIA JSON file
for js in list_VIA_JSON_files:
    coco_conversion(str(js))


# %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
# Combine all COCO JSON files

list_regen_COCO_files = [
    x
    for x in parent_dir_via_json_files.glob("**/*")
    if x.is_file() and str(x).endswith("_coco_regen.json")
]
list_regen_COCO_files.sort()


js0 = list_regen_COCO_files[0]
for j, jsf in enumerate(list_regen_COCO_files[1:], start=1):
    js_combined = combine_coco(str(js0), str(jsf))

    # update js0
    js0 = js_combined
