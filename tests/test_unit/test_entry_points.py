import os

import pytest


@pytest.mark.parametrize(
    "cli_command",
    ["extract-frames", "combine-annotations", "train-detector"],
)
def test_smoke(cli_command: str) -> None:
    status = os.system(f"{cli_command} --help")
    assert status == 0
