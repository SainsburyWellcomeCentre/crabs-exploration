import lightning as pl
import optuna

from crabs.detection_tracking.models import FasterRCNN


def objective(
    trial, config, data_module, accelerator, mlf_logger, fast_dev_run
):
    # with mlflow.start_run():
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

    print(config)
    print(fast_dev_run)

    # Initialize the model
    lightning_model = FasterRCNN(config)

    # Use the MLFlow logger instance passed from the main loop
    # mlf_logger.log_hyperparams(
    #     {f"learning_rate_trial_{trial.number}": learning_rate}
    # )
    # mlf_logger.log_hyperparams(
    #     {f"num_epochs_trial_{trial.number}": num_epochs}
    # )

    # Initialize the PyTorch Lightning Trainer
    trainer = pl.Trainer(
        max_epochs=config["num_epochs"],
        accelerator=accelerator,
        logger=mlf_logger,
        fast_dev_run=fast_dev_run,
    )

    # Train the model
    trainer.fit(lightning_model, data_module)

    # Evaluate the model on the test dataset
    # test_result = trainer.test(ckpt_path=None, datamodule=data_module)

    # # Log the evaluation metric
    # mlflow.log_metric(
    #     f"test_precision_trial_{trial.number}",
    #     test_result[0]["test_precision"],
    # )
    # Return the evaluation metric to optimize
    # return test_result[0]["val_precision"]
    # Log the validation precision from the last epoch
    val_precision = trainer.callback_metrics["val_precision"]

    # Return the validation precision to optimize
    return val_precision


def optimize_hyperparameters(
    config, data_module, accelerator, mlf_logger, fast_dev_run
):
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
    best_params = study.best_params
    print("Best hyperparameters:", best_params)

    return best_params
