import os

import pytest


@pytest.mark.parametrize(
    "cli_tool_name",
    [
        "extract-frames",
        "combine-and-format-annotations",
    ],
)
def test_cli_tool(cli_tool_name: str) -> None:
    status = os.system(f"{cli_tool_name} --help")
    assert status == 0
