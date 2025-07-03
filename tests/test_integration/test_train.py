import subprocess
from pathlib import Path

import cv2
import numpy as np
import pooch
import pytest


@pytest.fixture()
def input_data_paths(pooch_registry: pooch.Pooch, tmp_path: Path):
    """Input data for a detector+tracking run.

    The data is fetched from the pooch registry.
    """
    # Create a directory with black small frames under tmp_path
    black_frames_dir = tmp_path / "black_frames"
    black_frames_dir.mkdir(parents=True, exist_ok=False)
    for i in range(100):
        black_frame = np.zeros((10, 10, 3), dtype=np.uint8)
        cv2.imwrite(black_frames_dir / f"{i:08d}.jpg", black_frame)

    # Get an annotation file from pooch for 100 frames
    annotation_file = pooch_registry.fetch(
        "sample_train_data/annotations_100frames.json"
    )

    return {
        "dataset_dir": black_frames_dir,
        "annotation_file": annotation_file,
        "seed_n": "42",
        "accelerator": "cpu",
        "experiment_name": "test_train_detector",
        "limit_train_batches": "0.01",
    }


def test_train_detector(input_data_paths: dict, tmp_path: Path):
    # Define command
    main_command = [
        "train-detector",
        f"--dataset_dirs={input_data_paths['dataset_dir']}",
        f"--annotation_files={input_data_paths['annotation_file']}",
        f"--seed_n={input_data_paths['seed_n']}",
        f"--accelerator={input_data_paths['accelerator']}",
        f"--experiment_name={input_data_paths['experiment_name']}",
        "--log_data_augmentation",
        "--fast_dev_run",
        f"--limit_train_batches={input_data_paths['limit_train_batches']}",
    ]

    # Run command
    completed_process = subprocess.run(
        main_command,
        check=True,
        cwd=tmp_path,
        # set cwd to Pytest's temporary directory
        # so the output is saved there
    )

    # check the command runs successfully
    assert completed_process.returncode == 0

    # check mlflow folder is created

    # check data augmentation is logged

    # check model is trained?
