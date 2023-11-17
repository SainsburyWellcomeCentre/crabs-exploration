from pathlib import Path

import typer

from crabs.bboxes_labelling.annotations_utils import (
    combine_multiple_via_jsons,
    convert_via_json_to_coco,
)

# instantiate Typer app
app = typer.Typer(rich_markup_mode="rich")


@app.command()
def combine_VIA_and_convert_to_COC0(
    parent_dir_via_jsons: str,
    via_default_dir: str,
    via_project_name: str,
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

    # Convert to COCO and return path
    return convert_via_json_to_coco(json_out_fullpath)


def app_wrapper():
    app()


if __name__ == "__main__":
    app_wrapper()
