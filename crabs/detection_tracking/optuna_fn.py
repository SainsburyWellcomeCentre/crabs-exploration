import datetime
import optuna
import lightning as pl
from crabs.detection_tracking.datamodule import CustomDataModule
from crabs.detection_tracking.models import FasterRCNN


def objective(
    trial,
    config,
    main_dirs,
    annotation_files,
    accelerator,
    seed_n,
    experiment_name,
):
    # Sample hyperparameters from the search space
    learning_rate = trial.suggest_loguniform("learning_rate", 1e-6, 1e-4)
    num_epochs = trial.suggest_int("num_epochs", 1, 10)

    # Update the config with the sampled hyperparameters
    config["learning_rate"] = learning_rate
    config["num_epochs"] = num_epochs

    # Initialize the data module
    data_module = CustomDataModule(main_dirs, annotation_files, config, seed_n)

    # Initialize the model
    lightning_model = FasterRCNN(config)

    # Initialize the MLFlow logger
    timestamp = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    run_name = f"run_{timestamp}"
    mlf_logger = pl.pytorch.loggers.MLFlowLogger(
        run_name=run_name,
        experiment_name=experiment_name,
        tracking_uri="file:./ml-runs",
    )
    mlf_logger.log_hyperparams(config)

    # Initialize the PyTorch Lightning Trainer
    trainer = pl.Trainer(
        max_epochs=config["num_epochs"],
        accelerator=accelerator,
        logger=mlf_logger,
        # fast_dev_run=True,
    )

    # Train the model
    trainer.fit(
        lightning_model,
        data_module,
    )

    # Evaluate the model on the test dataset
    test_result = trainer.test(datamodule=data_module)

    # Return the evaluation metric to optimize
    return test_result[0]["test_precision"]


def optimize_hyperparameters(
    config, main_dirs, annotation_files, accelerator, seed_n, experiment_name
):
    # Create an Optuna study
    study = optuna.create_study(direction="minimize")

    # Define objective function with partial args
    objective_fn = lambda trial: objective(
        trial,
        config,
        main_dirs,
        annotation_files,
        accelerator,
        seed_n,
        experiment_name,
    )

    # Optimize the objective function
    study.optimize(objective_fn, n_trials=3)

    # Get the best hyperparameters
    best_params = study.best_params
    print("Best hyperparameters:", best_params)

    return best_params
