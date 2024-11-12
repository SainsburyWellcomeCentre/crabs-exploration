"""Pytest fixtures for integration tests."""

from datetime import datetime
from pathlib import Path

import pooch
import pytest

GIN_TEST_DATA_REPO = "https://gin.g-node.org/SainsburyWellcomeCentre/crabs-exploration-test-data"


# @pytest.fixture(autouse=True)
# def mock_home_directory(monkeypatch: pytest.MonkeyPatch):
#     """Monkeypatch pathlib.Path.home().

#     Instead of returning the usual home path, the
#     monkeypatched version returns the path to
#     Path.home() / ".mock-home". This
#     is to avoid local tests interfering with the
#     potentially existing user data on the same machine.

#     Parameters
#     ----------
#     monkeypatch : pytest.MonkeyPatch
#        a monkeypatch fixture

#     """
#     # define mock home path
#     home_path = Path.home()  # actual home path
#     mock_home_path = home_path / ".mock-home"

#     # create mock home directory if it doesn't exist
#     if not mock_home_path.exists():
#         mock_home_path.mkdir()

#     # monkeypatch Path.home() to point to the mock home
#     def mock_home():
#         return mock_home_path

#     monkeypatch.setattr(Path, "home", mock_home)


@pytest.fixture(scope="session")
def pooch_registry() -> dict:
    """Pooch registry for the test data.

    This fixture is common for all the test session. The
    file registry is downloaded fresh for every test session.

    Returns
    -------
    dict
        URL and hash of the GIN repository with the test data

    """
    # Use pytest fixture? Should it be wiped out after a session?
    # Initialise a pooch registry for the test data
    registry = pooch.create(
        Path.home() / ".crabs-exploration-test-data",
        base_url=f"{GIN_TEST_DATA_REPO}/raw/master/test_data",
    )

    # Download the registry file from GIN to the pooch cache
    # force to download it every time by using a timestamped file name
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    file_registry = pooch.retrieve(
        url=f"{GIN_TEST_DATA_REPO}/raw/master/files-registry.txt",
        known_hash=None,
        fname=f"files-registry_{timestamp}.txt",
        path=Path.home() / ".crabs-exploration-test-data",
    )

    # Load registry file onto pooch registry
    registry.load_registry(
        file_registry,
    )

    # Remove registry file
    Path(file_registry).unlink()

    return registry
