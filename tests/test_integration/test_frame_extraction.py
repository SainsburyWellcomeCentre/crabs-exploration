import datetime
import json
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


@pytest.fixture()
def cli_inputs_dict(tmp_path: Path) -> dict:
    """Returns the command line input arguments as a dictionary.

    These command line arguments are passed to the
    extract frames CLI command. The output path is
    set to Pytest's temporary directory to manage
    teardown.

    Parameters
    ----------
    tmp_path : Path
        Pytest fixture providing a temporary path

    Returns
    -------
    dict
        a dictionary with parameters for the frame extraction
    """
    return {
        "output-path": str(tmp_path),
        "video-extensions": ["mp4"],
        "initial-samples": 5,
        "scale": 0.5,
        "n-components": 3,
        "n-clusters": 5,
        "per-cluster": 1,
        "compute-features-per-video": True,
    }


@pytest.fixture()
def cli_inputs_list(cli_inputs_dict: dict) -> list:
    """Returns the command line input arguments as a list.

    Parameters
    ----------
    cli_inputs_dict : dict
        a dictionary with parameters for the frame extraction

    Returns
    -------
    list
        a list with parameters for the frame extraction
    """

    def cli_inputs_dict_to_list(input_params: dict) -> list:
        """Transforms a dictionary of parameters into a list of CLI arguments.

        If for an item in the dictionary its value is True,
        the key is taken as a CLI boolean argument (i.e., a flag).

        If for an item in the dictionary the value is False,
        its key with '--no-' prepended is taken as a CLI boolean argument
        (i.e., a flag). This matches typer behaviour.

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
        list_kys_modified = []
        for k in input_params.keys():
            if input_params[k] is False:
                list_kys_modified.append("--no-" + k)
            else:
                list_kys_modified.append("--" + k)

        list_cli_args = []
        for ky, val in zip(list_kys_modified, input_params.values()):
            if isinstance(val, list):
                list_cli_args.append(str(ky))
                for elem in val:
                    list_cli_args.append(str(elem))

            elif not isinstance(val, bool):
                for elem in [ky, val]:
                    list_cli_args.append(str(elem))

        return list_cli_args

    return cli_inputs_dict_to_list(cli_inputs_dict)


@pytest.fixture()
def video_extensions_flipped(input_data_dir: str) -> list:
    """Extracts the extensions of video files in input_data_dir
    and flips their case (uppercase -> lowercase and viceversa).

    The file extensions would be provided by the user in the
    typical use case.

    Parameters
    ----------
    input_data_dir : str
        path to the directory containing the video files

    Returns
    -------
    list
        list of file extensions
    """
    # build list of video files
    list_files = list_files_in_dir(input_data_dir)

    # get unique extensions for all files
    list_unique_extensions = list({f.suffix[1:] for f in list_files})

    # flip the case of the extensions
    list_extensions_flipped = [ext.lower() for ext in list_unique_extensions]
    list_extensions_flipped = list(set(list_extensions_flipped))

    return list_extensions_flipped


@pytest.fixture()
def mock_extract_frames_app(
    cli_inputs_dict: dict,
) -> typer.main.Typer:
    """Monkeypatches the extract-frames app to modify its default values.

    We modify the defaults with values that are more convenient for testing.

    Parameters
    ----------
    cli_inputs_dict : dict
        a dictionary with parameters for the frame extraction

    Returns
    -------
    typer.main.Typer
        an app with the same functionality as the one defined for
        `compute_and_extract_frames_to_label` but with different default values.
    """
    from crabs.bboxes_labelling.extract_frames_to_label_w_sleap import (
        compute_and_extract_frames_to_label,
    )

    # instantiate app
    app = typer.Typer(rich_markup_mode="rich")

    # link mocked command to app
    # change the defaults so that they match cli_inputs_dict
    @app.command()
    def mock_combine_and_format_annotations(
        list_video_locations: list[str],
        output_path: str = cli_inputs_dict["output-path"],
        output_subdir: Optional[str] = None,
        video_extensions: list[str] = cli_inputs_dict["video-extensions"],
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


def list_files_in_dir(input_dir: str) -> list:
    """Lists files in input directory

    Parameters
    ----------
    input_dir : str
        path to directory

    Returns
    -------
    list
        list of files in input directory
    """

    return [
        f
        for f in Path(input_dir).glob("*")
        if f.is_file() and not f.name.startswith(".")
    ]


def check_output_files(list_input_videos: list, cli_dict: dict) -> None:
    """Run assertions on output files from frame extraction

    Parameters
    ----------
    list_input_videos : list
        list of videos used for frame extraction
    cli_dict : dict
        a dictionary with the parameters of the frame extraction
    """
    # check name of directory with extracted frames
    list_subfolders = [
        f.path for f in os.scandir(cli_dict["output-path"]) if f.is_dir()
    ]
    extracted_frames_dir = Path(list_subfolders[0])
    assert len(list_subfolders) == 1
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

    if cli_dict["compute-features-per-video"]:
        n_expected_imgs = (
            cli_dict["n-clusters"]
            * cli_dict["per-cluster"]
            * len(list_input_videos)
        )
    else:
        n_expected_imgs = cli_dict["n-clusters"] * cli_dict["per-cluster"]

    assert len(list_imgs) <= n_expected_imgs

    # check filename format of images: <video_name>_frame_{frame_idx:08d}
    list_regex_patterns = [
        Path(input_video_str).stem + "_frame_[\d]{8}$"
        for input_video_str in list_input_videos
    ]
    for f in list_imgs:
        assert (
            sum(
                [
                    bool(re.fullmatch(regex, f.stem))
                    for regex in list_regex_patterns
                ]
            )
            == 1
        )  # only one must match

    # check n_elements in json file matches n of files generated
    with open((extracted_frames_dir / "extracted_frames.json")) as js:
        extracted_frames_dict = json.load(js)
        n_extracted_frames = sum(
            [len(list_idcs) for list_idcs in extracted_frames_dict.values()]
        )
        assert n_extracted_frames == len(list_imgs)


@pytest.mark.parametrize(
    "input_video",
    [
        "NINJAV_S001_S001_T003_subclip_p1_05s.mp4",
        "NINJAV_S001_S001_T003_subclip_p2_05s.MP4",
    ],
)
def test_frame_extraction_one_video(
    input_video: str,
    input_data_dir: str,
    cli_inputs_list: list,
    cli_inputs_dict: dict,
) -> None:
    """Test frame extraction on one video

    Parameters
    ----------
    input_video : str
        input video filename
    input_data_dir : str
        path to input video directory
    cli_inputs_list : list
        command line input arguments for frame extraction as a list
    cli_inputs_dict : dict
        command line input arguments as a dictionary, for validation
    """
    # import app
    from crabs.bboxes_labelling.extract_frames_to_label_w_sleap import app

    # invoke app
    runner = CliRunner()
    input_video_path = str(Path(input_data_dir) / input_video)
    result = runner.invoke(app, args=[input_video_path] + cli_inputs_list)

    # check exit code
    assert result.exit_code == 0

    # check output files
    check_output_files([input_video_path], cli_inputs_dict)


@pytest.mark.parametrize(
    "input_video",
    [
        "NINJAV_S001_S001_T003_subclip_p1_05s.mp4",
        "NINJAV_S001_S001_T003_subclip_p2_05s.MP4",
    ],
)
def test_frame_extraction_one_video_defaults(
    input_video: str,
    input_data_dir: str,
    cli_inputs_dict: dict,
    mock_extract_frames_app: typer.main.Typer,
) -> None:
    """Test frame extraction on one video, using default CLI arguments

    Parameters
    ----------
    input_video : str
        input video filename
    input_data_dir : str
        path to input video directory
    cli_inputs_dict : dict
        command line input arguments as a dictionary, for validation
    mock_extract_frames_app: typer.main.Typer
        a monkeypatched app with convenient defaults for testing
    """
    # import mocked app
    app = mock_extract_frames_app

    # call mocked app
    runner = CliRunner()
    input_video_path = str(Path(input_data_dir) / input_video)
    result = runner.invoke(app, args=input_video_path)
    assert result.exit_code == 0

    # check output files
    check_output_files([input_video_path], cli_inputs_dict)


def test_frame_extraction_one_dir(
    input_data_dir: str,
    cli_inputs_list: list,
    cli_inputs_dict: dict,
) -> None:
    """Test frame extraction on one input directory.

    Frames are extracted from all video files in the input
    directory.

    Parameters
    ----------
    input_data_dir : str
        path to input video directory
    cli_inputs_list : list
        command line input arguments for frame extraction as a list
    cli_inputs_dict : dict
        command line input arguments as a dictionary, for validation
    """
    # import app
    from crabs.bboxes_labelling.extract_frames_to_label_w_sleap import app

    # invoke app
    runner = CliRunner()
    result = runner.invoke(app, args=[input_data_dir] + cli_inputs_list)

    # check exit code
    assert result.exit_code == 0

    # check files
    # list of input videos
    list_input_videos = list_files_in_dir(input_data_dir)
    list_input_videos = [
        f
        for f in list_input_videos
        if any(
            [
                str(f).lower().endswith(ext)
                for ext in cli_inputs_dict["video-extensions"]
            ]
        )
    ]
    check_output_files(list_input_videos, cli_inputs_dict)


def test_frame_extraction_one_dir_defaults(
    input_data_dir: str,
    cli_inputs_dict: dict,
    mock_extract_frames_app: typer.main.Typer,
) -> None:
    """Test frame extraction on one input directory, using default
    CLI arguments.

    Frames are extracted from all video files in the input
    directory.

    Parameters
    ----------
    input_data_dir : str
        path to input video directory
    cli_inputs_dict : dict
        command line input arguments as a dictionary, for validation
    mock_extract_frames_app : typer.main.Typer
        a monkeypatched app with convenient defaults for testing.
    """
    # import mock app
    app = mock_extract_frames_app

    # invoke app
    runner = CliRunner()
    result = runner.invoke(app, args=input_data_dir)

    # check exit code
    assert result.exit_code == 0

    # check files
    # list of input videos
    list_input_videos = list_files_in_dir(input_data_dir)
    list_input_videos = [
        f
        for f in list_input_videos
        if any(
            [
                str(f).lower().endswith(ext)
                for ext in cli_inputs_dict["video-extensions"]
            ]
        )
    ]
    check_output_files(list_input_videos, cli_inputs_dict)


def test_extension_case_insensitive(
    input_data_dir: str, video_extensions_flipped: list
) -> None:
    """
    Tests that the function that computes the list of SLEAP videos
    is case-insensitive for the user-provided extension.

    Parameters
    ----------
    input_video_dir : pathlib.Path
        path to the input video directory
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
