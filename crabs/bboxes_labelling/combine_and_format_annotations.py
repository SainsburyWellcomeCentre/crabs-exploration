"""Script to combine and format annotations."""

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
    exclude_pattern: str | None = None,
    via_default_dir: str | None = None,
    via_project_name: str | None = None,
) -> str:
    r"""Combine a list of VIA JSON files into one and convert to COCO format.

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
    all_files = Path(parent_dir_via_jsons).glob("*")
    list_input_json_files = [
        x for x in all_files if x.is_file() and str(x).endswith(".json")
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
    """Wrap function for the Typer app."""
    app()


if __name__ == "__main__":
    app_wrapper()
