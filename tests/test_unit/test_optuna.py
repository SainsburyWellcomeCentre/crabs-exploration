import argparse
from unittest.mock import MagicMock, patch

import pytest

from crabs.detection_tracking.datamodules import CrabsDataModule
from crabs.detection_tracking.optuna_fn import optimize_hyperparameters


@pytest.fixture
def config():
    return {
        "num_classes": [2],
        "optuna_param": {
            "n_trials": [3],
            "learning_rate": [1e-6, 1e-4],
            "num_epochs": [1, 2],
        },
        "checkpoint_saving": {},
    }


@pytest.fixture
def data_module():
    return MagicMock(spec=CrabsDataModule)


@pytest.fixture
def args():
    return argparse.Namespace()


def mock_objective(
    trial,
    config,
    data_module,
    accelerator,
    fast_dev_run,
    limit_train_batches,
    experiment_name,
    mlflow_folder,
    args,
):
    # Simulate usage of trial parameters
    learning_rate = trial.suggest_float(
        "learning_rate",
        config["optuna_param"]["learning_rate"][0],
        config["optuna_param"]["learning_rate"][1],
    )
    num_epochs = trial.suggest_int(
        "num_epochs",
        config["optuna_param"]["num_epochs"][0],
        config["optuna_param"]["num_epochs"][1],
    )
    return learning_rate * num_epochs


def test_optimize_hyperparameters(config, data_module, args):
    with patch(
        "crabs.detection_tracking.optuna_fn.objective",
        side_effect=mock_objective,
    ):
        # Call the function
        result = optimize_hyperparameters(
            config=config,
            data_module=data_module,
            accelerator="cpu",
            experiment_name="test_experiment",
            mlflow_folder="/tmp/mlflow",
            args=args,
            fast_dev_run=True,
            limit_train_batches=False,
        )

        assert isinstance(result, dict), "Result should be a dictionary"
        assert (
            "n_trials" in config["optuna_param"]
        ), "n_trials should be in config"
        assert result, "Result dictionary should not be empty"
        assert "learning_rate" in result, "Result should contain learning_rate"
        assert "num_epochs" in result, "Result should contain num_epochs"
