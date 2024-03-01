from pathlib import Path
from typing import Optional

import pytest
import typer


@pytest.fixture()
def cli_inputs_dict(tmp_path: Path) -> dict:
    """Returns the command line input arguments as a dictionary.

    These command line arguments are passed to the
    extract frames CLI command. The output path is
    set to Pytest's temporary directory to manage
    teardown.
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
    """Returns the command line input arguments as a list."""

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
def cli_inputs_list_empty():
    return []


@pytest.fixture()
def mock_extract_frames_app(
    cli_inputs_dict: dict,
) -> typer.main.Typer:
    """Monkeypatches the extract-frames app to modify its default values.

    We modify the defaults with values that are more convenient for testing.
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
