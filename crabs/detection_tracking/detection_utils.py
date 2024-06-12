import argparse
import datetime
import os
from pathlib import Path
from typing import Any, Dict, Optional

from lightning.pytorch.loggers import MLFlowLogger

DEFAULT_ANNOTATIONS_FILENAME = "VIA_JSON_combined_coco_gen.json"


def prep_img_directories(dataset_dirs: list[str]) -> list[str]:
    """
    Derive list of input image directories from a list of dataset directories.
    We assume a specific structure for the dataset directories.

    Parameters:
    -----------
    dataset_dirs : List[str]
        List of directories containing dataset folders.

    Returns:
    --------
    List[str]:
        List of directories containing image frames.
    """
    images_dirs = []
    for dataset in dataset_dirs:
        images_dirs.append(str(Path(dataset) / "frames"))
    return images_dirs


def prep_annotation_files(
    input_annotation_files: list[str], dataset_dirs: list[str]
) -> list[str]:
    """
    Prepares annotation files for processing.

    Parameters:
    -----------
    input_annotation_files : List[str]
        List of annotation files or filenames.
    dataset_dirs : List[str]
        List of directories containing dataset folders.

    Returns:
    --------
    List[str]:
        List of annotation file paths.
    """
    # prepare list of annotation files
    annotation_files = []

    # if none are passed: assume default filename for annotations,
    # and default location under `annotations` directory
    if not input_annotation_files:
        for dataset in dataset_dirs:
            annotation_files.append(
                str(
                    Path(dataset)
                    / "annotations"
                    / DEFAULT_ANNOTATIONS_FILENAME
                )
            )

    # if a list of annotation files/filepaths is passed
    else:
        for annot, dataset in zip(input_annotation_files, dataset_dirs):
            # if the annotation is only filename:
            # assume file is under 'annotation' directory
            if Path(annot).name == annot:
                annotation_files.append(
                    str(Path(dataset) / "annotations" / annot)
                )
            # otherwise assume the full path to the annotations file is passed
            else:
                annotation_files.append(annot)

    return annotation_files


def set_mlflow_run_name() -> str:
    """
    Set MLflow run name.

    Use the slurm job ID if it is a SLURM job, else use a timestamp.
    For SLURM jobs:
    - if it is a single job use <job_ID>, else
    - if it is an array job use <job_ID_parent>_<task_ID>
    """
    # Get slurm environment variables
    slurm_job_id = os.environ.get("SLURM_JOB_ID")
    slurm_array_job_id = os.environ.get("SLURM_ARRAY_JOB_ID")

    # If job is a slurm array job
    if slurm_job_id and slurm_array_job_id:
        slurm_task_id = os.environ.get("SLURM_ARRAY_TASK_ID")
        run_name = f"run_slurm_{slurm_array_job_id}_{slurm_task_id}"
    # If job is a slurm single job
    elif slurm_job_id:
        run_name = f"run_slurm_{slurm_job_id}"
    # If not a slurm job: use timestamp
    else:
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        run_name = f"run_{timestamp}"

    return run_name


def setup_mlflow_logger(
    experiment_name: str,
    run_name: str,
    mlflow_folder: str,
    ckpt_config: dict = {},
) -> MLFlowLogger:
    """
    Setup MLflow logger for a given experiment and run name. If a
    checkpointing config is passed, it will setup the logger with a
    checkpointing callback.

    Parameters
    ----------
    experiment_name : str
        Name of the experiment under which this run will be logged.
    run_name : str
        Name of the run.
    mlflow_folder : str
        Path to folder where to store MLflow outputs for this run.
    ckpt_config : dict
        A dictionary with the checkpointing parameters. By default, an empty dict.

    Returns
    -------
    MLFlowLogger
        A logger to record data for MLflow
    """

    # Setup logger with checkpointing
    mlf_logger = MLFlowLogger(
        experiment_name=experiment_name,
        run_name=run_name,
        tracking_uri=f"file:{Path(mlflow_folder)}",
        log_model=ckpt_config.get("copy_as_mlflow_artifacts", False),
    )

    return mlf_logger


def log_metadata_to_logger(
    mlf_logger: MLFlowLogger,
    cli_args: argparse.Namespace,
) -> MLFlowLogger:
    """
    Log metadata to MLflow logger.

    Add CLI arguments and, if available, SLURM job information.

    Parameters
    ----------
    mlf_logger : MLFlowLogger
        An MLflow logger instance.
    cli_args : argparse.Namespace
        Parsed command-line arguments.

    Returns
    -------
    MLFlowLogger
        An MLflow logger instance with metadata logged.
    """

    # Log CLI arguments
    mlf_logger.log_hyperparams({"cli_args": cli_args})

    # Log slurm metadata
    slurm_job_id = os.environ.get("SLURM_JOB_ID")
    slurm_array_job_id = os.environ.get("SLURM_ARRAY_JOB_ID")

    # if array job
    if slurm_job_id and slurm_array_job_id:
        slurm_task_id = os.environ.get("SLURM_ARRAY_TASK_ID")
        mlf_logger.log_hyperparams(
            {"slurm_job_id": slurm_array_job_id}
        )  # ID of parent job
        mlf_logger.log_hyperparams({"slurm_array_task_id": slurm_task_id})

    # if single job
    elif slurm_job_id:
        mlf_logger.log_hyperparams({"slurm_job_id": slurm_job_id})

    return mlf_logger


def setup_logger(
    experiment_name: str,
    mlflow_folder: str,
    config_ckpt: Dict[str, Any],
    args: argparse.Namespace,
    run_name: Optional[str] = None,
) -> MLFlowLogger:
    """
    Setup MLflow logger for training, with checkpointing.

    Includes logging metadata about the job (CLI arguments and SLURM job IDs).

    Parameters
    ----------
    experiment_name : str
        Name of the experiment under which this run will be logged.
    mlflow_folder : str
        Path to the folder where MLflow outputs for this run will be stored.
    config_ckpt : dict
        Dictionary containing the checkpointing parameters.
    args : dict
        Dictionary of CLI arguments.
    run_name : str, optional
        Name of the run. By default, the run name will be generated. This option is for optuna

    Returns
    -------
    MLFlowLogger
        MLFlowLogger instance.
    """
    # Assign run name
    if run_name is None:
        run_name = set_mlflow_run_name()

    # Setup logger with checkpointing
    mlf_logger = setup_mlflow_logger(
        experiment_name=experiment_name,
        run_name=run_name,
        mlflow_folder=mlflow_folder,
        ckpt_config=config_ckpt,
    )

    # # Log metadata: CLI arguments and SLURM (if required)
    mlf_logger = log_metadata_to_logger(mlf_logger, args)

    # Log (assumed) path to checkpoints directory
    print(type(Path(mlf_logger._tracking_uri)))
    print(type(mlf_logger._experiment_id))
    print(type(mlf_logger._run_id))
    print(type("checkpoint"))

    path_to_checkpoints = (
        Path(mlf_logger._tracking_uri)
        / str(mlf_logger._experiment_id)
        / str(mlf_logger._run_id)
        / "checkpoints"
    )
    mlf_logger.log_hyperparams(
        {"path_to_checkpoints": str(path_to_checkpoints)}
    )

    return mlf_logger


def slurm_logs_as_artifacts(logger: MLFlowLogger, slurm_job_id: str) -> None:
    """
    Add SLURM logs as MLflow artifacts of the current run.

    This function assumes the filenaming convention from the training scripts
    located at `crabs-exploration/bash_scripts/`.

    Parameters
    ----------
    logger : MLFlowLogger
        An MLflow logger instance.
    slurm_job_id : str
        The SLURM job ID for the current run.

    Returns
    -------
    None
    """

    # Get slurm env variables: slurm and array job ID
    slurm_node = os.environ.get("SLURMD_NODENAME")
    slurm_array_job_id = os.environ.get("SLURM_ARRAY_JOB_ID")

    # Get root of log filenames
    # for array job
    if slurm_array_job_id:
        slurm_task_id = os.environ.get("SLURM_ARRAY_TASK_ID")
        log_filename = (
            f"slurm_array.{slurm_array_job_id}-{slurm_task_id}.{slurm_node}"
        )
    # for single job
    else:
        log_filename = f"slurm.{slurm_job_id}.{slurm_node}"

    # Add log files as artifacts of this run
    for ext in ["out", "err"]:
        logger.experiment.log_artifact(
            logger.run_id,
            f"{log_filename}.{ext}",
        )
