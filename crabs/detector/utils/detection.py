"""Utils used in training and evaluation."""

import argparse
import datetime
import os
from pathlib import Path
from typing import Any, Optional

import torch
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
    cli_args: argparse.Namespace,
    ckpt_config: dict[str, Any] = {},
) -> MLFlowLogger:
    """
    Setup MLflow logger and log job metadata, with optional checkpointing.

    Setup MLflow logger for a given experiment and run name. If a
    checkpointing config is passed, it will setup the logger with a
    checkpointing callback. A job metadata is the job's CLI arguments and
    the SLURM job IDs (if relevant).

    Parameters
    ----------
    experiment_name : str
        Name of the experiment under which this run will be logged.
    run_name : str
        Name of the run.
    mlflow_folder : str
        Path to folder where to store MLflow outputs for this run.
    cli_args : argparse.Namespace
        Parsed command-line arguments.
    ckpt_config : dict
        A dictionary with the checkpointing parameters.
        By default, an empty dict.

    Returns
    -------
    MLFlowLogger
        A logger to record data for MLflow
    """

    # Setup MLflow logger for a given experiment and run name
    # (with checkpointing if required)
    mlf_logger = MLFlowLogger(
        experiment_name=experiment_name,
        run_name=run_name,
        tracking_uri=f"file:{Path(mlflow_folder)}",
        log_model=ckpt_config.get("copy_as_mlflow_artifacts", False),
    )

    # Log metadata: CLI arguments and SLURM job ID (if required)
    mlf_logger = log_metadata_to_logger(mlf_logger, cli_args)

    # If checkpointing is not an empty dict,
    # log the path to the checkpoints directory
    if ckpt_config:
        path_to_checkpoints = (
            Path(mlf_logger._tracking_uri)
            / mlf_logger._experiment_id
            / mlf_logger._run_id
            / "checkpoints"
        )
        mlf_logger.log_hyperparams(
            {"path_to_checkpoints": str(path_to_checkpoints)}
        )

    return mlf_logger


def slurm_logs_as_artifacts(logger: MLFlowLogger, slurm_job_id: str):
    """
    Add slurm logs as an MLflow artifacts of the current run.
    The filenaming convention from the training scripts at crabs-exploration/bash_scripts/ is assumed.
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


def bbox_tensors_to_COCO_dict(
    bbox_tensors: torch.Tensor, list_img_filenames: Optional[list] = None
) -> dict:
    """Convert list of bounding boxes as tensors to COCO-crab format.

    Parameters
    ----------
    bbox_tensors : list[torch.Tensor]
        List of tensors with bounding boxes for each image.
        Each element of the list corresponds to an image, and each tensor in
        the list contains the bounding boxes for that image. Each tensor is of
        size (n, 4) where n is the number of bounding boxes in the image.
        The 4 values in the second dimension are x_min, y_min, x_max, y_max.
    list_img_filenames : list[str], optional
        List of image filenames. If not provided, filenames are generated
        as "frame_{i:04d}.png" where i is the 0-based index of the image in the
        list of bounding boxes.

    Returns
    -------
    dict
        COCO format dictionary with bounding boxes.
    """
    # Create list of image filenames if not provided
    if list_img_filenames is None:
        list_img_filenames = [
            f"frame_{i:04d}.png" for i in range(len(bbox_tensors))
        ]

    # Create list of dictionaries for images
    list_images: list[dict] = []
    for img_id, img_name in enumerate(list_img_filenames):
        image_entry = {
            "id": img_id + 1,  # 1-based
            "width": 0,
            "height": 0,
            "file_name": img_name,
        }
        list_images.append(image_entry)

    # Create list of dictionaries for annotations
    list_annotations: list[dict] = []
    for img_id, img_bboxes in enumerate(bbox_tensors):
        # loop thru bboxes in image
        for bbox_row in img_bboxes:
            x_min, y_min, x_max, y_max = bbox_row.numpy().tolist()
            # we convert the array to list to make it JSON serializable

            annotation = {
                "id": len(list_annotations) + 1,  # 1-based
                "image_id": img_id,
                "category_id": 1,
                "bbox": [x_min, y_min, x_max - x_min, y_max - y_min],
                "area": (x_max - x_min) * (y_max - y_min),
                "iscrowd": 0,
            }

            list_annotations.append(annotation)

    # Create COCO dictionary
    coco_dict = {
        "info": {},
        "licenses": [],
        "categories": [{"id": 1, "name": "crab", "supercategory": "animal"}],
        "images": list_images,
        "annotations": list_annotations,
    }

    return coco_dict
