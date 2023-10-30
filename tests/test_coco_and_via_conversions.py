import datetime
import os
from pathlib import Path

import pytest

from bboxes_labelling.annotations_utils import (
    combine_all_via_jsons,
    convert_via_json_to_coco,
    read_json_file,
)


@pytest.fixture()
def via_json_1() -> str:
    """Return path to a sample VIA JSON file.

    Returns
    -------
    str
        path to a sample VIA JSON file 1.
    """
    # Return path to sample VIA (Visual Image Annotator) JSON file 1
    return str(Path("tests/data/COCO_VIA_JSONS/VIA_JSON_1.json").resolve())


@pytest.fixture()
def via_json_2() -> str:
    """Return path to a sample VIA JSON file.

    Returns
    -------
    str
        path to a sample VIA JSON file 2.
    """
    # Return path to sample VIA JSON file 2
    return str(Path("tests/data/COCO_VIA_JSONS/VIA_JSON_2.json").resolve())


def test_via_json_combine(
    via_json_1: str,
    via_json_2: str,
    tmp_path: Path,
) -> None:
    """Test combination of VIA JSON files.

    Check if the combination of 2 VIA JSON files has the same data as the
    individual JSONS.

    Parameters
    ----------
    via_json_1 : str
        path to a sample VIA JSON file 1.
    via_json_2 : str
        path to a sample VIA JSON file 2.
    tmp_path : Path
        Pytest fixture providing a temporary directory path
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
        json_out_dir=str(tmp_path),
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


def test_via_json_combine_default_dir(
    via_json_1: str,
    via_json_2: str,
    tmp_path: Path,
) -> None:
    """Test combination of VIA JSON files adding a default VIA directory.

    Check if the combination of 2 VIA JSON files has the specified default
    VIA directory.

    Parameters
    ----------
    via_json_1 : str
        path to a sample VIA JSON file 1.
    via_json_2 : str
        path to a sample VIA JSON file 2.
    tmp_path : Path
        Pytest fixture providing a temporary directory path
    """
    # Set default directory
    via_default_dir = "/sample/VIA/project/directory"

    # Combine JSONs 1 and 2
    timestamp_str = datetime.datetime.now().strftime("%d%m%Y_%H%M%S")
    via_json_combined = combine_all_via_jsons(
        [via_json_1, via_json_2],
        json_out_filename=f"VIA_JSON_combined_{timestamp_str}.json",
        json_out_dir=str(tmp_path),
        via_default_dir=via_default_dir,
    )
    # Read combination as dict
    via_json_combined_dict = read_json_file(via_json_combined)

    # Check default directory is as specified
    if not via_default_dir.endswith(os.sep):
        via_default_dir = f"{via_default_dir}{os.sep}"
        # add trailing slash if required
    assert (
        via_json_combined_dict["_via_settings"]["core"]["default_filepath"]
        == via_default_dir
    )


def test_via_json_combine_project_name(
    via_json_1: str,
    via_json_2: str,
    tmp_path: Path,
) -> None:
    """Test combination of VIA JSON files adding a VIA project name.

    Check if the combination of 2 VIA JSON files has the specified project
    name.

    Parameters
    ----------
    via_json_1 : str
        path to a sample VIA JSON file 1.
    via_json_2 : str
        path to a sample VIA JSON file 2.
    tmp_path : Path
        Pytest fixture providing a temporary directory path
    """
    # Set project name
    via_project_name = "TEST"

    # Combine JSONs 1 and 2
    timestamp_str = datetime.datetime.now().strftime("%d%m%Y_%H%M%S")
    via_json_combined = combine_all_via_jsons(
        [via_json_1, via_json_2],
        json_out_filename=f"VIA_JSON_combined_{timestamp_str}.json",
        json_out_dir=str(tmp_path),
        via_project_name=via_project_name,
    )
    # read combination as dict
    via_json_combined_dict = read_json_file(via_json_combined)

    # check project name is as specified
    assert (
        via_json_combined_dict["_via_settings"]["project"]["name"] == via_project_name
    )


@pytest.mark.parametrize(
    "via_json_file",
    ["via_json_1", "via_json_2"],
)
def test_coco_generated_from_via_json(
    via_json_file: str,
    tmp_path: Path,
    request: pytest.FixtureRequest,
) -> None:
    """Test COCO file generated from VIA JSON is correct.

    Check if the COCO file generated from the VIA JSON contains the same data.
    We assume all annotations are the same COCO category.

    Parameters
    ----------
    via_json_file : str
        path to a sample VIA JSON file
    tmp_path : Path
        Pytest fixture providing a temporary directory path
    request: pytest.FixtureRequest
        to request a parametrized fixture
    """
    # Define category attributes of the annotations
    coco_category_ID = 1
    coco_category_name = "crab"
    coco_supercategory_name = "animal"

    # Convert via_json_file to COCO
    coco_out_fullpath = convert_via_json_to_coco(
        request.getfixturevalue(via_json_file),
        coco_out_dir=str(tmp_path),
    )

    # Load dictitionaries for VIA JSON file and COCO generated file
    via_json_dict = read_json_file(request.getfixturevalue(via_json_file))
    coco_json_dict = read_json_file(coco_out_fullpath)

    # Check COCO categories are as expected
    assert len(coco_json_dict["categories"]) == 1
    assert coco_json_dict["categories"][0]["id"] == coco_category_ID
    assert coco_json_dict["categories"][0]["name"] == coco_category_name
    assert coco_json_dict["categories"][0]["supercategory"] == coco_supercategory_name

    # We assume keys in VIA's metadata list are sorted in increasing image ID
    # (ID starts from 1)
    list_keys_via_img_metadata = list(
        via_json_dict["_via_img_metadata"].keys(),
    )  # keys will be in insertion order,
    # see https://docs.python.org/3/tutorial/datastructures.html#dictionaries

    # Compare images' IDs and filenames between VIA JSON and COCO file
    for img_dict in coco_json_dict["images"]:
        ky_in_via_img_metadata = list_keys_via_img_metadata[img_dict["id"] - 1]

        # Check key in VIA image metadata matches COCO filename
        assert Path(img_dict["file_name"]).stem == Path(ky_in_via_img_metadata).stem

        # Check VIA filename matches COCO filename
        assert (
            img_dict["file_name"]
            == via_json_dict["_via_img_metadata"][ky_in_via_img_metadata]["filename"]
        )

    # Compare annotations between VIA JSON and COCO file
    # Note:
    # - img_id and annotation_id are in increasing order
    #   in coco_json_1_dict["annotations"]
    # - img_id and annotation_id start from 1
    img_id_previous_image = 1
    ann_idx_per_img = 0
    for annotation_dict in coco_json_dict["annotations"]:
        # Get image id for this annotation from COCO
        img_id = annotation_dict["image_id"]

        # Get image info from VIA using the "image_id" from the COCO annotation
        ky_in_via_img_metadata = list_keys_via_img_metadata[img_id - 1]
        img_dict_in_via = via_json_dict["_via_img_metadata"][ky_in_via_img_metadata]

        # Reset annotation index per image if image_id changes
        if img_id_previous_image != img_id:
            ann_idx_per_img = 0
            img_id_previous_image = img_id

        # Get shape of this annotation from VIA
        reg = img_dict_in_via["regions"][ann_idx_per_img]
        w_from_via = reg["shape_attributes"]["width"]
        h_from_via = reg["shape_attributes"]["height"]
        bbox_from_via = [
            reg["shape_attributes"]["x"],
            reg["shape_attributes"]["y"],
            w_from_via,
            h_from_via,
        ]

        # Check key in VIA image metadata matches filename derived from
        # COCO annotation
        img_name = coco_json_dict["images"][img_id - 1]["file_name"]
        assert Path(ky_in_via_img_metadata).stem == Path(img_name).stem

        # Check annotations are all the same category
        assert annotation_dict["category_id"] == 1

        # Check annotations are all "iscrowd"=0
        assert annotation_dict["iscrowd"] == 0

        # Check bounding box shape and area matches
        assert annotation_dict["bbox"] == bbox_from_via
        assert annotation_dict["area"] == w_from_via * h_from_via

        # Update annotation index for next iteration
        ann_idx_per_img += 1
