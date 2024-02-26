import os

import pytest


@pytest.mark.parametrize(
    "cli_command",
    [
        "extract-frames",
        "combine-annotations",
    ],
)
def test_entry_points(cli_tool_name: str) -> None:
    status = os.system(f"{cli_tool_name} --help")
    assert status == 0
