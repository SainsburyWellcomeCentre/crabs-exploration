import datetime
import os
import re
from pathlib import Path
from typing import Optional

import pytest
import typer
from typer.testing import CliRunner

from crabs.bboxes_labelling.extract_frames_to_label_w_sleap import (
    get_list_of_sleap_videos,
)


@pytest.fixture()
def input_data_dir() -> str:
    return str(Path(__file__).parents[1] / "data" / "clips")


@pytest.fixture(
    params=[
        "NINJAV_S001_S001_T003_subclip_p1_05s.mp4",
        "NINJAV_S001_S001_T003_subclip_p2_05s.mp4",
    ]
)
def input_video(input_data_dir, request) -> str:
    return str(Path(input_data_dir) / request.param)


@pytest.fixture()
def cli_inputs_dict(tmp_path: Path) -> dict:
    """These are also the default values

    Parameters
    ----------
    tmp_path : Path
        _description_

    Returns
    -------
    dict
        _description_
    """
    return {
        "output-path": str(tmp_path),
        "video-extensions": "mp4",
        "initial-samples": 5,
        "scale": 0.5,
        "n-components": 3,
        "n-clusters": 5,
        "per-cluster": 1,
        "compute-features-per-video": "",
    }


@pytest.fixture()
def cli_inputs_list(cli_inputs_dict: dict) -> list:
    def cli_inputs_dict_to_list(input_params: dict) -> list:
        """Transforms a dictionary of parameters into a list of CLI arguments
        that can be passed to `subprocess.run()`.

        If for an item in the dictionary the value is empty string,
        its key is taken as a CLI boolean argument (i.e., a flag).

        Parameters
        ----------
        input_params : dict
            dictionary with the command line arguments to transform.
            If a value is an empty string, the corresponding key is
            considered a boolean input argument (i.e., a flag).

        Returns
        -------
        list
            a list of command line arguments to pass to `subprocess.run()`.
        """

        list_kys_modified = ["--" + k for k in input_params.keys()]
        list_cli_args = [
            str(elem)
            for pair in zip(list_kys_modified, input_params.values())
            for elem in pair
            if elem != ""
        ]

        return list_cli_args

    return cli_inputs_dict_to_list(cli_inputs_dict)


@pytest.fixture()
def video_extensions_flipped(input_data_dir: str) -> list:
    # build list of video files
    list_files = list_files_in_dir(input_data_dir)

    # get unique extensions for all files
    list_unique_extensions = list({f.suffix[1:] for f in list_files})

    # flip the case of the extensions
    list_user_extensions_flip = [ext.lower() for ext in list_unique_extensions]
    list_user_extensions_flip = list(set(list_user_extensions_flip))

    return list_user_extensions_flip


@pytest.fixture()
def mock_extract_frames_to_label_w_sleap(
    cli_inputs_dict: dict,
) -> typer.main.Typer:
    from crabs.bboxes_labelling.extract_frames_to_label_w_sleap import (
        compute_and_extract_frames_to_label,
    )

    # instantiate app
    app = typer.Typer(rich_markup_mode="rich")

    # link mocked command
    # we change the defaults so that they match cli_inputs_dict
    @app.command()
    def mock_combine_and_format_annotations(
        list_video_locations: list[str],
        output_path: str = cli_inputs_dict["output-path"],
        output_subdir: Optional[str] = None,
        video_extensions: list[str] = [cli_inputs_dict["video-extensions"]],
        initial_samples: int = cli_inputs_dict["initial-samples"],
        sample_method: str = "stride",
        scale: float = cli_inputs_dict["scale"],
        feature_type: str = "raw",
        n_components: int = cli_inputs_dict["n-components"],
        n_clusters: int = cli_inputs_dict["n-clusters"],
        per_cluster: int = cli_inputs_dict["per-cluster"],
        compute_features_per_video: bool = True,
    ):
        return compute_and_extract_frames_to_label(
            list_video_locations,
            output_path=output_path,
            output_subdir=output_subdir,
            video_extensions=video_extensions,
            initial_samples=initial_samples,
            sample_method=sample_method,
            scale=scale,
            feature_type=feature_type,
            n_components=n_components,
            n_clusters=n_clusters,
            per_cluster=per_cluster,
            compute_features_per_video=compute_features_per_video,
        )

    return app


def list_files_in_dir(input_dir: str):
    # build list of files in dir
    return [
        f
        for f in Path(input_dir).glob("*")
        if f.is_file() and not f.name.startswith(".")
    ]


def check_output_files(input_video_str: str, cli_dict: dict):
    # check name of directory with extracted frames
    list_subfolders_with_paths = [
        f.path for f in os.scandir(cli_dict["output-path"]) if f.is_dir()
    ]

    extracted_frames_dir = Path(list_subfolders_with_paths[0])

    assert len(list_subfolders_with_paths) == 1
    assert (
        type(
            datetime.datetime.strptime(
                extracted_frames_dir.name, "%Y%m%d_%H%M%S"
            )
        )
        == datetime.datetime
    )

    # check there is an extracted_frames.json file
    assert (extracted_frames_dir / "extracted_frames.json").is_file()

    # check min number of image files
    # NOTE: total number of files generated is not deterministic
    list_imgs = [f for f in extracted_frames_dir.glob("*.png")]
    assert (
        len(list_imgs) <= cli_dict["n-clusters"] * cli_dict["per-cluster"]
    )  # per video?

    # check format of images filename
    regex_pattern = Path(input_video_str).stem + "_frame_[\d]{8}$"
    assert all([re.fullmatch(regex_pattern, f.stem) for f in list_imgs])

    # check n_elements in json file matches n of files generated


def test_frame_extraction_one_video(
    input_video: str,
    cli_inputs_list: list,
    cli_inputs_dict: dict,
):
    # import app
    from crabs.bboxes_labelling.extract_frames_to_label_w_sleap import app

    # invoke app
    runner = CliRunner()
    result = runner.invoke(app, args=[input_video] + cli_inputs_list)

    # check exit code
    assert result.exit_code == 0

    # check output files
    check_output_files(input_video, cli_inputs_dict)


def test_frame_extraction_one_video_defaults(
    input_video: str,
    cli_inputs_dict: dict,
    mock_extract_frames_to_label_w_sleap: typer.main.Typer,
):
    # import mocked app
    app = mock_extract_frames_to_label_w_sleap

    # call mocked app
    runner = CliRunner()
    result = runner.invoke(app, args=input_video)
    assert result.exit_code == 0

    # check output files
    check_output_files(input_video, cli_inputs_dict)


def test_frame_extraction_one_dir(
    input_data_dir: str,
    cli_inputs_list: list,
):
    # import app
    from crabs.bboxes_labelling.extract_frames_to_label_w_sleap import app

    # invoke app
    runner = CliRunner()
    result = runner.invoke(app, args=[input_data_dir] + cli_inputs_list)

    # check exit code
    assert result.exit_code == 0

    # check files


def test_frame_extraction_one_dir_defaults(
    input_data_dir: str, mock_extract_frames_to_label_w_sleap
):
    # import mock app
    app = mock_extract_frames_to_label_w_sleap

    # invoke app
    runner = CliRunner()
    result = runner.invoke(app, args=input_data_dir)

    # check exit code
    assert result.exit_code == 0

    # check files


def test_extension_case_insensitive(
    input_data_dir: str, video_extensions_flipped: list
):
    """
    Tests that the function that computes the list of SLEAP videos
    is case-insensitive for the user-input extension.

    Parameters
    ----------
    input_video_dir : pathlib.Path
        Path to the input video directory
    """

    # build list of video files in dir
    list_files = list_files_in_dir(input_data_dir)

    # compute list of SLEAP videos for the given user extensions;
    # the extensions are passed with the opposite case as the file extensions
    list_sleap_videos = get_list_of_sleap_videos(
        [input_data_dir],
        video_extensions_flipped,
    )

    # check list of SLEAP videos matches the list of files
    assert len(list_sleap_videos) == len(list_files)
    assert len(list_sleap_videos) == len(list_files)
