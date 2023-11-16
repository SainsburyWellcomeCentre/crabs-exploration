import os

import pytest


@pytest.mark.parametrize("cli_tool_name", ["extract-frames"])
def test_cli_args(cli_tool_name: str) -> None:
    status = os.system(f"{cli_tool_name} --help")
    assert status == 0
