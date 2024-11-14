"""Pytest fixtures for integration tests."""

from datetime import datetime
from pathlib import Path

import pooch
import pytest

GIN_TEST_DATA_REPO = "https://gin.g-node.org/SainsburyWellcomeCentre/crabs-exploration-test-data"


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
    # Initialise pooch registry
    registry = pooch.create(
        Path.home() / ".crabs-exploration-test-data",
        base_url=f"{GIN_TEST_DATA_REPO}/raw/master/test_data",
    )

    # Download the registry file from GIN
    # Force to download it fresh every time by using a timestamped filename
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

    # Delete registry file
    Path(file_registry).unlink()

    return registry
