import argparse
import datetime
import os
from typing import Any, Dict

import lightning
import optuna
from optuna.trial import Trial

from crabs.detection_tracking.datamodules import CrabsDataModule
from crabs.detection_tracking.detection_utils import (
    setup_logger,
    slurm_logs_as_artifacts,
)
from crabs.detection_tracking.models import FasterRCNN


def objective(
    trial: Trial,
    config: Dict,
    data_module: CrabsDataModule,
    accelerator: str,
    fast_dev_run: bool,
    limit_train_batches: bool,
    experiment_name: str,
    mlflow_folder: str,
    args,
) -> float:
    """
    Objective function for Optuna optimization.

    Parameters
    ----------
    trial: Trial
        Optuna trial object.
    config: Dict
        Configuration dictionary containing hyperparameters search space.
    data_module: DataModule
        Data module for loading training and validation data.
    accelerator: str
        PyTorch Lightning accelerator.
    fast_dev_run: bool
        Flag to run a fast development version.

    Returns
    -------
    float
        Validation precision achieved by the model.
    """
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    run_name = f"run_{trial.number}_{timestamp}"

    # Get MLflow logger
    mlf_logger = setup_logger(
        experiment_name,
        mlflow_folder,
        config.get("checkpoint_saving", {}),
        args,
        run_name,
    )

    # Sample hyperparameters from the search space
    learning_rate = trial.suggest_float(
        "learning_rate",
        float(config["optuna"]["learning_rate"][0]),
        float(config["optuna"]["learning_rate"][1]),
    )
    num_epochs = trial.suggest_int(
        "num_epochs",
        config["optuna"]["n_epochs"][0],
        config["optuna"]["n_epochs"][1],
    )

    # Log the sampled hyperparameters
    mlf_logger.log_hyperparams(
        {
            "trial_number": trial.number,
            "learning_rate": learning_rate,
            "num_epochs": num_epochs,
        }
    )

    # Update the config with the sampled hyperparameters
    config["learning_rate"] = learning_rate
    config["n_epochs"] = num_epochs

    # Initialize the model
    lightning_model = FasterRCNN(config)

    # Initialize the PyTorch Lightning Trainer
    trainer = lightning.Trainer(
        max_epochs=config["num_epochs"],
        accelerator=accelerator,
        logger=mlf_logger,
        fast_dev_run=fast_dev_run,
        limit_train_batches=limit_train_batches,
    )

    # Train the model
    trainer.fit(lightning_model, data_module)

    # if this is a slurm job: add slurm logs as artifacts
    slurm_job_id = os.environ.get("SLURM_JOB_ID")
    if slurm_job_id:
        slurm_logs_as_artifacts(trainer.logger, slurm_job_id)

    val_precision = trainer.callback_metrics["val_precision"].item()
    val_recall = trainer.callback_metrics["val_recall"].item()
    avg_val_value = (val_precision + val_recall) / 2

    # Log metrics
    mlf_logger.log_metrics(
        {
            "val_precision": val_precision,
            "val_recall": val_recall,
            "avg_val_value": avg_val_value,
        }
    )

    return avg_val_value


def optimize_hyperparameters(
    config: Dict,
    data_module: CrabsDataModule,
    accelerator: str,
    experiment_name: str,
    mlflow_folder: str,
    args: argparse.Namespace,
    fast_dev_run: bool,
    limit_train_batches: bool,
) -> Dict[str, Any]:
    """
    Optimize hyperparameters using Optuna.

    Parameters
    ----------
    config: Dict
        Configuration dictionary containing hyperparameters search space.
    data_module: DataModule
        Data module for loading training and validation data.
    accelerator: str
        PyTorch Lightning accelerator.
    fast_dev_run: bool
        Flag to run a fast development version.

    Returns
    -------
    Dict
        Best hyperparameters found by the optimization.
    """
    # Create an Optuna study
    study = optuna.create_study(direction="maximize")

    # Define objective function with partial args
    def objective_fn(trial):
        return objective(
            trial,
            config,
            data_module,
            accelerator,
            fast_dev_run,
            limit_train_batches,
            experiment_name,
            mlflow_folder,
            args,
        )

    # Optimize the objective function
    study.optimize(objective_fn, n_trials=config["optuna"]["n_trials"])

    # Get the best hyperparameters
    best_trial = study.best_trial
    best_params = best_trial.params

    print("Best hyperparameters:", best_params)

    return best_params
