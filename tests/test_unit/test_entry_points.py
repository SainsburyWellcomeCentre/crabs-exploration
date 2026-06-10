import os

import pytest


@pytest.mark.parametrize(
    "cli_command",
    [
        "extract-frames",
        "combine-annotations",
        "train-detector",
        "evaluate-detector",
        "detect-and-track-video",
        "create-zarr-dataset",
        "compute-burrow-prompt-frames",
        "compute-burrow-prompt-coords",
    ],
)
def test_smoke(cli_command: str) -> None:
    status = os.system(f"{cli_command} --help")
    assert status == 0
