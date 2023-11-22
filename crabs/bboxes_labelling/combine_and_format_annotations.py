import re
from pathlib import Path

import typer

from crabs.bboxes_labelling.annotations_utils import (
    combine_multiple_via_jsons,
    convert_via_json_to_coco,
)

# instantiate Typer app
app = typer.Typer(rich_markup_mode="rich")


@app.command()
def combine_VIA_and_convert_to_COCO(
    parent_dir_via_jsons: str,
    via_default_dir: str,
    via_project_name: str,
    exclude_pattern=None,
) -> str:
    """Combine a list of VIA JSON files into one and convert to COCO format

    Parameters
    ----------
    parent_dir_via_jsons : str
        path to the parent directory containing VIA JSON files
    via_default_dir : str
        The default directory in which to look for images for the VIA project.
        A full path is required.
    via_project_name : str
        The name of the VIA project.

    Returns
    -------
    str
        path to the COCO json file. By default, the file
    """

    # Get list of all JSON files
    list_json_files = [
        x
        for x in Path(parent_dir_via_jsons).glob("*")
        if x.is_file() and str(x).endswith(".json")
    ]

    # Exclude pattern if required
    if exclude_pattern:
        list_selected_json_files = [
            js
            for js in list_json_files
            if not re.search(exclude_pattern, str(js))
        ]
    else:
        list_selected_json_files = list_json_files.copy()

    list_selected_json_files.sort()

    # Combine VIA JSONS
    json_out_fullpath = combine_multiple_via_jsons(
        list_selected_json_files,
        via_default_dir=via_default_dir,
        via_project_name=via_project_name,
    )

    # Convert to COCO and return path
    return convert_via_json_to_coco(json_out_fullpath)


def app_wrapper():
    app()


if __name__ == "__main__":
    app_wrapper()
