import subprocess
from pathlib import Path

import cv2
import numpy as np
import pooch
import pytest
import yaml


@pytest.fixture()
def input_data_paths(pooch_registry: pooch.Pooch, tmp_path: Path):
    """Input data for a detector+tracking run.

    The data is fetched from the pooch registry.
    """
    # Create a sample dataset directory with black small frames at tmp_path
    # (note the package assumes the frames are in a subdirectory
    # called "frames")
    black_frames_dir = tmp_path / "frames"
    black_frames_dir.mkdir(parents=True, exist_ok=False)
    for i in range(100):
        black_frame = np.zeros((10, 10, 3), dtype=np.uint8)
        cv2.imwrite(str(black_frames_dir / f"{i:08d}.png"), black_frame)

    # Get an annotation file from pooch for 100 frames
    annotation_file = pooch_registry.fetch(
        "sample_train_data/annotations_100frames.json"
    )

    # Read the reference config file
    reference_config_file = (
        Path(__file__).parents[2] / "crabs/detector/config/faster_rcnn.yaml"
    )
    with open(reference_config_file) as f:
        reference_config = yaml.safe_load(f)

    # Modify the config file to have num_workers = 0 and
    # save it at tmp_path
    # This is to avoid issues with macos-15-intel CI tests failing with
    # "RuntimeError: Please call `iter(combined_loader)` first"
    # when num_workers > 0, Python 3.9 and macos-15-intel.
    # https://github.com/pytorch/pytorch/issues/46648
    reference_config["num_workers"] = 0
    config_file = tmp_path / "config.yaml"
    with open(config_file, "w") as f:
        yaml.dump(reference_config, f)

    return {
        "dataset_dir": black_frames_dir.parent,
        "annotation_file": annotation_file,
        "config_file": config_file,
        "seed_n": "42",
        "accelerator": "cpu",
        "experiment_name": "test_train_detector",
        "limit_train_batches": "0.1",
    }


def test_train_detector(input_data_paths: dict, tmp_path: Path):
    # Define command
    main_command = [
        "train-detector",
        f"--dataset_dirs={input_data_paths['dataset_dir']}",
        f"--annotation_files={input_data_paths['annotation_file']}",
        f"--config_file={input_data_paths['config_file']}",
        f"--seed_n={input_data_paths['seed_n']}",
        f"--accelerator={input_data_paths['accelerator']}",
        f"--experiment_name={input_data_paths['experiment_name']}",
        "--log_data_augmentation",
        "--fast_dev_run",  # suppresses logging and checkpointing
        f"--limit_train_batches={input_data_paths['limit_train_batches']}",
    ]

    # Run command
    completed_process = subprocess.run(
        main_command,
        check=True,
        cwd=tmp_path,
        text=True,
        capture_output=True,
        # set cwd to Pytest's temporary directory
        # so the output is saved there
    )

    # check the command runs successfully
    assert completed_process.returncode == 0

    # check mlflow folder is created
    assert (tmp_path / "ml-runs").exists()

    # check the training stopped early
    assert (
        "`Trainer.fit` stopped: `max_steps=1` reached."
        in completed_process.stderr
    )
