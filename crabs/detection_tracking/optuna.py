from typing import Callable, Dict

import optuna


def compute_optimal_hyperparameters(
    objective_fn: Callable, config_optuna: Dict, direction: str = "maximize"
) -> dict:
    """Compute hyperparameters that optimize the objective function.

    Use Optuna to optimize the objective function and return the
    hyperparameters that maximise or minimise the objective function.

    Parameters
    ----------
    objective_fn : Callable
        A function that takes `trial` as input and returns the value to
        maximise or minimise (a float).
    config_optuna : Dict
        A dictionary with the configuration parameters for Optuna.
    direction : str, optional
        Specifies whether the optimisation is to maximise or minimise
        the objective function. By default "maximize".

    Returns
    -------
    dict
        The optimal parameters computed by Optuna.
    """
    # Create an study
    study = optuna.create_study(direction=direction)

    # Optimize the objective function
    study.optimize(
        objective_fn,  #
        n_trials=config_optuna["n_trials"],
    )

    # Extract results
    best_trial = study.best_trial  # Should we log this?
    best_hyperparameters = best_trial.params

    return best_hyperparameters


def convert_string_number(num_s: str) -> int | float:
    """Convert a string to a float or an integer."""
    try:
        num = float(num_s)
        if num.is_integer():
            return int(num)
        return num
    except ValueError:
        raise ValueError(
            f"The provided string '{num_s}' is not a valid number"
        )
