import argparse
from unittest.mock import MagicMock, patch

import pytest

from crabs.detector.train_model import DectectorTrain
from crabs.detector.utils.hpo import compute_optimal_hyperparameters


@pytest.fixture
def config():
    return {
        "num_classes": [2],
        "optuna": {
            "n_trials": 3,
            "learning_rate": [1e-6, 1e-4],
        },
        "checkpoint_saving": {},
    }


@pytest.fixture
def args():
    return argparse.Namespace(
        config_file="dummy_config.yaml",
        dataset_dirs=["path/to/images"],
        annotation_files=["path/to/annotations"],
        seed_n=42,
        accelerator="cpu",
        experiment_name="test_experiment",
        mlflow_folder="/tmp/mlflow",
        fast_dev_run=True,
        limit_train_batches=False,
        checkpoint_path=None,
    )


@pytest.fixture
def detector_train(args, config):
    with patch.object(DectectorTrain, "load_config_yaml", MagicMock()):
        train_instance = DectectorTrain(args=args)
        print(config)
        train_instance.config = config
        return train_instance


def mock_core_training():
    return MagicMock(
        callback_metrics={
            "val_precision_optuna": MagicMock(item=lambda: 0.8),
            "val_recall_optuna": MagicMock(item=lambda: 0.7),
        }
    )


def test_optimize_hyperparameters(detector_train, config):
    config_optuna = config["optuna"]

    with patch.object(
        detector_train, "core_training", side_effect=mock_core_training
    ):
        result = compute_optimal_hyperparameters(
            objective_fn=detector_train.optuna_objective_fn,
            config_optuna=config_optuna,
            direction="maximize",
        )

        assert isinstance(result, dict), "Result should be a dictionary"
        assert (
            "n_trials" in config_optuna
        ), "n_trials should be in config_optuna"
        assert result, "Result dictionary should not be empty"
        assert "learning_rate" in result, "Result should contain learning_rate"
