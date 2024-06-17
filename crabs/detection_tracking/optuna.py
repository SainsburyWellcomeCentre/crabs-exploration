from typing import Callable, Dict

import optuna


def compute_optimal_hyperparameters(
    objective_fn: Callable, config_optuna: Dict, direction: str = "maximize"
):
    # Create an Optuna study
    study = optuna.create_study(direction=direction)

    # Optimize the objective function
    study.optimize(
        objective_fn,  # takes trial, returns float
        n_trials=config_optuna["n_trials"][0],  # why 0?
    )

    # Extract optimal results
    best_trial = study.best_trial
    best_hyperparameters = best_trial.params

    return best_hyperparameters
