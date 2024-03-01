from pathlib import Path
from typing import Optional

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
    exclude_pattern: Optional[str] = None,
    via_default_dir: Optional[str] = None,
    via_project_name: Optional[str] = None,
) -> str:
    """Combine a list of VIA JSON files into one and convert to COCO format

    Parameters
    ----------
    parent_dir_via_jsons : str
        path to the parent directory containing VIA JSON files
    exclude_pattern : Optional[str], optional
        a regex pattern that matches files to exclude.
        By default, None.  E.g.: "\w+_coco_gen.json$"
    via_default_dir : str
        The default directory in which to look for images for the VIA project.
        If None, the value specified in the first VIA JSON file is used.
        If a path is provided it needs to be a full path.
    via_project_name : str
        The name of the VIA project.
        If None, the value specified in the first VIA JSON file is used.

    Returns
    -------
    str
        path to the COCO json file. By default, the file
    """

    # Get list of all JSON files in directory
    list_input_json_files = [
        x
        for x in Path(parent_dir_via_jsons).glob("*")
        if x.is_file() and str(x).endswith(".json")
    ]

    # Combine VIA JSON files (excluding those with pattern if required)
    json_out_fullpath = combine_multiple_via_jsons(
        list_input_json_files,
        exclude_pattern=exclude_pattern,
        via_default_dir=via_default_dir,
        via_project_name=via_project_name,
    )

    # Convert to COCO and return path
    return convert_via_json_to_coco(json_out_fullpath)


def app_wrapper():
    app()


if __name__ == "__main__":
    app_wrapper()
