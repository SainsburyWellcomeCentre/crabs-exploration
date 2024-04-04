from typing import Any, Dict

import lightning as pl
import optuna
from optuna.trial import Trial

from crabs.detection_tracking.datamodules import CrabsDataModule
from crabs.detection_tracking.models import FasterRCNN


def objective(
    trial: Trial,
    config: Dict,
    data_module: CrabsDataModule,
    accelerator: str,
    mlf_logger: Any,
    fast_dev_run: bool,
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
    mlf_logger : Any
        MLflow logger object.
    fast_dev_run: bool
        Flag to run a fast development version.

    Returns
    -------
    float
        Validation precision achieved by the model.
    """
    # Sample hyperparameters from the search space
    learning_rate = trial.suggest_float(
        "learning_rate",
        float(config["optuna_param"]["learning_rate"][0]),
        float(config["optuna_param"]["learning_rate"][1]),
    )
    num_epochs = trial.suggest_int(
        "num_epochs",
        config["optuna_param"]["num_epochs"][0],
        config["optuna_param"]["num_epochs"][1],
    )

    # Update the config with the sampled hyperparameters
    config["learning_rate"] = learning_rate
    config["num_epochs"] = num_epochs

    # Initialize the model
    lightning_model = FasterRCNN(config)

    # Initialize the PyTorch Lightning Trainer
    trainer = pl.Trainer(
        max_epochs=config["num_epochs"],
        accelerator=accelerator,
        logger=mlf_logger,
        fast_dev_run=fast_dev_run,
    )

    # Train the model
    trainer.fit(lightning_model, data_module)

    val_precision = trainer.callback_metrics["val_precision"].item()
    # Log val_precision to MLflow
    trial.set_user_attr("val_precision", val_precision)

    return val_precision


def optimize_hyperparameters(
    config: Dict,
    data_module: CrabsDataModule,
    accelerator: str,
    mlf_logger: Any,
    fast_dev_run: bool,
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
    mlf_logger : Any
        MLflow logger object.
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
            trial, config, data_module, accelerator, mlf_logger, fast_dev_run
        )

    # Optimize the objective function
    study.optimize(
        objective_fn, n_trials=config["optuna_param"]["n_trials"][0]
    )

    # Get the best hyperparameters
    best_trial = study.best_trial
    best_params = best_trial.params
    best_val_precision = best_trial.user_attrs["val_precision"]

    print("Best hyperparameters:", best_params)
    print("Best val_precision:", best_val_precision)
    mlf_logger.log_hyperparams(best_params)
    mlf_logger.log_hyperparams({"best_val_precision": best_val_precision})

    return best_params
