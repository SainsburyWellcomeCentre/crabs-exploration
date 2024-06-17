from typing import Callable, Dict

import optuna


def compute_optimal_hyperparameters(
    objective_fn: Callable, config_optuna: Dict, direction: str = "maximize"
):
    # Create an Optuna study
    study = optuna.create_study(direction=direction)

    # Optimize the objective function
    study.optimize(
        objective_fn,  # a function that takes trial and returns a float
        n_trials=config_optuna["n_trials"],
    )

    # Extract optimal results
    best_trial = study.best_trial
    best_hyperparameters = best_trial.params

    return best_hyperparameters
