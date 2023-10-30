# Test

import datetime
import os
from pathlib import Path

import pytest

from bboxes_labelling.annotations_utils import (
    combine_all_via_jsons,
    read_json_file,
)


@pytest.fixture()
def via_json_1():
    """_summary_.

    Returns
    -------
    _type_
        _description_
    """
    # Return path to sample VIA (Visual Image Annotator) JSON file 1
    return str(Path("tests/data/COCO_VIA_JSONS/VIA_JSON_1.json").resolve())


@pytest.fixture()
def via_json_2():
    """_summary_.

    Returns
    -------
    _type_
        _description_
    """
    # Return path to sample VIA JSON file 2
    return str(Path("tests/data/COCO_VIA_JSONS/VIA_JSON_2.json").resolve())


@pytest.fixture()
def via_default_dir():
    """_summary_.

    Returns
    -------
    _type_
        _description_
    """
    # Return path to VIA project directory
    return "/sample/VIA/project/directory"


@pytest.fixture()
def via_project_name():
    """_summary_.

    Returns
    -------
    _type_
        _description_
    """
    # Return VIA project name
    return "TEST"


def test_via_json_combine(via_json_1, via_json_2, tmpdir):
    """_summary_.

    Returns
    -------
    _type_
        _description_
    """
    # Check if the combination of 2 VIA JSON files has the same data
    # as the separate JSONS

    # load sample JSONs 1 and 2 files as dicts
    via_json_1_dict = read_json_file(via_json_1)
    via_json_2_dict = read_json_file(via_json_2)

    # combine JSONs 1 and 2
    timestamp_str = datetime.datetime.now().strftime("%d%m%Y_%H%M%S")
    via_json_combined = combine_all_via_jsons(
        [via_json_1, via_json_2],
        json_out_filename=f"VIA_JSON_combined_{timestamp_str}.json",
        json_out_dir=tmpdir,
    )
    # read combination as dict
    via_json_combined_dict = read_json_file(via_json_combined)

    # check values taken from the first file to combine:
    # VIA settings
    assert via_json_combined_dict["_via_settings"] == via_json_1_dict["_via_settings"]
    # VIA attributes
    assert (
        via_json_combined_dict["_via_attributes"] == via_json_1_dict["_via_attributes"]
    )
    # data format version
    assert (
        via_json_combined_dict["_via_data_format_version"]
        == via_json_1_dict["_via_data_format_version"]
    )

    # Check image metadata in JSON 1 and 2 exist in combined JSON and
    # their contents are the same
    for img_metadata_dict in [
        via_json_1_dict["_via_img_metadata"],
        via_json_2_dict["_via_img_metadata"],
    ]:
        for ky in img_metadata_dict:
            assert (
                img_metadata_dict[ky] == via_json_combined_dict["_via_img_metadata"][ky]
            )

    # Check the number of images in the combined JSON is the sum of the number
    # of images in JSON 1 and JSON 2
    assert len(via_json_1_dict["_via_img_metadata"].keys()) + len(
        via_json_2_dict["_via_img_metadata"].keys(),
    ) == len(via_json_combined_dict["_via_img_metadata"].keys())

    # Check image IDs from VIA_JSON_1 and 2 exist in combined
    for img_id_list in [
        via_json_1_dict["_via_image_id_list"],
        via_json_2_dict["_via_image_id_list"],
    ]:
        assert all(
            [x in via_json_combined_dict["_via_image_id_list"] for x in img_id_list],
        )


def test_via_json_combine_default_dir(via_json_1, via_json_2, via_default_dir, tmpdir):
    # Check if the combination of 2 VIA JSON files has the specified default
    # dir

    # combine JSONs 1 and 2
    timestamp_str = datetime.datetime.now().strftime("%d%m%Y_%H%M%S")
    via_json_combined = combine_all_via_jsons(
        [via_json_1, via_json_2],
        json_out_filename=f"VIA_JSON_combined_{timestamp_str}.json",
        json_out_dir=tmpdir,
        via_default_dir=via_default_dir,
        # via_project_name=via_project_name,
    )
    # read combination as dict
    via_json_combined_dict = read_json_file(via_json_combined)

    # check default directory is as specified
    if not via_default_dir.endswith(os.sep):
        via_default_dir = f"{via_default_dir}{os.sep}"
        # add trailing slash if required
    assert (
        via_json_combined_dict["_via_settings"]["core"]["default_filepath"]
        == via_default_dir
    )


def test_via_json_combine_project_name(
    via_json_1,
    via_json_2,
    via_project_name,
    tmpdir,
):
    # Check if the combination of 2 VIA JSON files has the specified project
    # name

    # combine JSONs 1 and 2
    timestamp_str = datetime.datetime.now().strftime("%d%m%Y_%H%M%S")
    via_json_combined = combine_all_via_jsons(
        [via_json_1, via_json_2],
        json_out_filename=f"VIA_JSON_combined_{timestamp_str}.json",
        json_out_dir=tmpdir,
        via_project_name=via_project_name,
    )
    # read combination as dict
    via_json_combined_dict = read_json_file(via_json_combined)

    # check project name is as specified
    assert (
        via_json_combined_dict["_via_settings"]["project"]["name"] == via_project_name
    )


# def test_coco_generated_from_via_json():
#     # Check if the COCO file generated from the VIA JSON contains the same data
