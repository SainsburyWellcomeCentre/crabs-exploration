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
    # Initialise pooch registry
    registry = pooch.create(
        Path.home() / ".crabs-exploration-test-data",
        base_url=f"{GIN_TEST_DATA_REPO}/raw/master/test_data",
    )

    # Download only the registry file from GIN
    # if known_hash = None, the file is always downloaded.
    file_registry = pooch.retrieve(
        url=f"{GIN_TEST_DATA_REPO}/raw/master/files-registry.txt",
        known_hash=None,
        fname="files-registry.txt",  
        # we need to pass a filename otherwise the file is not overwritten every time!
        # From the docs: if fname=None, will create a unique file name using 
        # a combination of the last part of the URL (assuming it’s the file 
        # name) and the MD5 hash of the URL. 
        path=Path.home() / ".crabs-exploration-test-data",
    )

    # Load registry file onto pooch registry
    registry.load_registry(file_registry)

    return registry
