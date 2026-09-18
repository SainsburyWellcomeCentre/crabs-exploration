"""Pytest fixtures for integration tests."""

from pathlib import Path

import pooch
import pytest

GIN_TEST_DATA_REPO = "https://gin.g-node.org/SainsburyWellcomeCentre/crabs-exploration-test-data"


@pytest.fixture(scope="session")
def pooch_registry() -> dict:
    """Pooch registry for the test data.

    This fixture is common to the entire test session. The
    file registry is downloaded fresh for every test session.

    Returns
    -------
    dict
        URL and hash of the GIN repository with the test data

    """
    # Cache the test data in the user's home directory
    test_data_dir = Path.home() / ".crabs-exploration-test-data"

    # Remove the file registry if it exists
    # otherwise the registry is not downloaded everytime
    file_registry_path = test_data_dir / "files-registry.txt"
    if file_registry_path.is_file():
        Path(file_registry_path).unlink()

    # Initialise pooch registry
    registry = pooch.create(
        test_data_dir,
        base_url=f"{GIN_TEST_DATA_REPO}/raw/master/test_data",
    )

    # Download only the registry file from GIN
    # (this file should always be downloaded fresh from GIN)
    file_registry = pooch.retrieve(
        url=f"{GIN_TEST_DATA_REPO}/raw/master/files-registry.txt",
        known_hash=None,
        fname=file_registry_path.name,
        path=file_registry_path.parent,
    )

    # Load registry file onto pooch registry
    registry.load_registry(file_registry)

    return registry
