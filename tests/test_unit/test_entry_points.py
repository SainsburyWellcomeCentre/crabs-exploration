import os

import pytest


@pytest.mark.parametrize(
    "cli_command",
    [
        "extract-frames",
        "extract-loops",
        "combine-annotations",
        "train-detector",
        "evaluate-detector",
        "detect-and-track-video",
        "extract-loops",
        "create-zarr-dataset",
    ],
)
def test_smoke(cli_command: str) -> None:
    status = os.system(f"{cli_command} --help")
    assert status == 0
