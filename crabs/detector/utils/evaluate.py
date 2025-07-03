"""Utils used in evaluation."""

import argparse
import ast
import sys
from pathlib import Path
from typing import Optional

import numpy as np
import torch
import yaml  # type: ignore
from scipy.optimize import linear_sum_assignment
from torchvision import ops

from crabs.detector.utils.detection import (
    prep_annotation_files,
    prep_img_directories,
)


def compute_precision_recall(
    pred_dicts_batch: list, gt_dicts_batch: list, iou_threshold: float
) -> tuple[float, float]:
    """Compute precision and recall.

    Parameters
    ----------
    pred_dicts_batch : list
        A list of prediction dictionaries for each element in the batch
        with keys 'boxes', 'labels', and 'scores'
    gt_dicts_batch : list
        A list of ground truth dictionaries for each element in the batch
        with keys 'image_id', 'boxes', 'labels'.
    iou_threshold : float
        IoU threshold for considering a detection as true positive

    Returns
    -------
    Tuple[float, float]
        precision and recall

    """
    # evaluate detections using hungarian algorithm
    eval_results = evaluate_detections_hungarian(
        pred_dicts_batch, gt_dicts_batch, iou_threshold
    )

    # compute precision and recall with division by zero handling
    total_detections = eval_results["tp"] + eval_results["fp"]
    total_gts = eval_results["tp"] + eval_results["fn"]

    precision = (
        (eval_results["tp"] / total_detections)
        if total_detections > 0
        else 0.0
    )
    recall = (eval_results["tp"] / total_gts) if total_gts > 0 else 0.0

    return precision, recall


def evaluate_detections_hungarian(
    pred_dicts_batch: list, gt_dicts_batch: list, iou_threshold: float
) -> dict:
    """Evaluate detection performance using Hungarian algorithm for matching.

    Parameters
    ----------
    pred_dicts_batch : list
        A list of prediction dictionaries for each element in the batch
        with keys 'boxes', 'labels', and 'scores'. Note that only the
        boxes are used for evaluation, not the labels or scores.
    gt_dicts_batch : list
        A list of ground truth dictionaries for each element in the batch
        with keys 'image_id', 'boxes', 'labels'. Note that only the
        boxes are used for evaluation, not the labels or the image_id.
    iou_threshold : float
        IoU threshold for considering a detection as true positive

    Returns
    -------
    dict
        Dictionary with keys "tp", "fp", and "fn"
        - tp: number of true positives
        - fp: number of false positives
        - fn: number of missed detections (false negatives)

    """
    # Concatenate detections and ground truth boxes across the batch
    pred_bboxes = torch.cat(
        [pred_dict["boxes"] for pred_dict in pred_dicts_batch]
    )
    gt_bboxes = torch.cat([gt_dict["boxes"] for gt_dict in gt_dicts_batch])

    # Initialize output arrays
    true_positives = np.zeros(len(pred_bboxes), dtype=bool)
    false_positives = np.zeros(len(pred_bboxes), dtype=bool)
    matched_gts = np.zeros(len(gt_bboxes), dtype=bool)
    missed_detections = np.zeros(len(gt_bboxes), dtype=bool)  # unmatched gts

    if len(pred_bboxes) > 0 and len(gt_bboxes) > 0:
        # Compute IoU matrix (pred_bboxes x gt_bboxes)
        iou_matrix = ops.box_iou(pred_bboxes, gt_bboxes).cpu().numpy()

        # Use Hungarian algorithm to find optimal assignment
        pred_indices, gt_indices = linear_sum_assignment(
            iou_matrix, maximize=True
        )

        # Mark true positives and false positives based on optimal assignment
        for pred_idx, gt_idx in zip(pred_indices, gt_indices, strict=True):
            if iou_matrix[pred_idx, gt_idx] > iou_threshold:
                true_positives[pred_idx] = True
                matched_gts[gt_idx] = True
            else:
                false_positives[pred_idx] = True

        # Mark unmatched predictions as false positives
        false_positives[~true_positives] = True

        # Mark unmatched ground truth as missed detections
        missed_detections[~matched_gts] = True

    elif len(pred_bboxes) == 0 and len(gt_bboxes) > 0:
        # No predictions, all ground truth are missed
        missed_detections[:] = True
    elif len(pred_bboxes) > 0 and len(gt_bboxes) == 0:
        # No ground truth, all predictions are false positives
        false_positives[:] = True

    # Return sum as a dict
    return {
        "tp": true_positives.sum().item(),
        "fp": false_positives.sum().item(),
        "fn": missed_detections.sum().item(),
    }


def get_mlflow_parameters_from_ckpt(trained_model_path: str) -> dict:
    """Get MLflow client from ckpt path and associated params."""
    from mlflow.tracking import MlflowClient

    # roughly assert the format of the path is correct
    # Note: to check if this is an MLflow chekcpoint,
    # we simply check if the parent directory is called
    # 'checkpoints', so it is not a very strict check.
    try:
        assert (
            Path(trained_model_path).parent.stem == "checkpoints"
        ), "The parent directory to an MLflow checkpoint is expected to be called 'checkpoints'"  # noqa: E501
    except AssertionError as e:
        print(f"Assertion failed: {e}")
        sys.exit(1)

    # get mlruns path, experiment and run ID associated to this checkpoint
    ckpt_mlruns_path = str(Path(trained_model_path).parents[3])
    # ckpt_experimentID = Path(trained_model_path).parents[2].stem
    ckpt_runID = Path(trained_model_path).parents[1].stem

    # create an Mlflow client to interface with mlflow runs
    mlrun_client = MlflowClient(
        tracking_uri=ckpt_mlruns_path,
    )

    # get parameters of the run
    run = mlrun_client.get_run(ckpt_runID)
    params = run.data.params
    params["run_name"] = run.info.run_name

    return params


def get_config_from_ckpt(
    config_file: Optional[str], trained_model_path: str
) -> dict:
    """Get config from checkpoint if config is not passed as a CLI argument."""
    # If config in CLI arguments: used passed config
    if config_file:
        with open(config_file) as f:
            config_dict = yaml.safe_load(f)

    # If not: used config from ckpt
    else:
        params = get_mlflow_parameters_from_ckpt(
            trained_model_path
        )  # string-dict

        # create a 1-level dict
        config_dict = {}
        for p in params:
            if p.startswith("config"):
                config_dict[p.replace("config/", "")] = ast.literal_eval(
                    params[p]
                )

        # format as a 2-levels nested dict
        # forward slashes in a key indicate a nested dict
        for key in list(config_dict):  # list makes a copy of original keys
            if "/" in key:
                key_parts = key.split("/")
                assert len(key_parts) == 2
                if key_parts[0] not in config_dict:
                    config_dict[key_parts[0]] = {
                        key_parts[1]: config_dict.pop(key)
                    }
                else:
                    config_dict[key_parts[0]].update(
                        {key_parts[1]: config_dict.pop(key)}
                    )

        # check there are no more levels
        assert all(["/" not in key for key in config_dict])

    return config_dict


def get_cli_arg_from_ckpt(
    args: argparse.Namespace, cli_arg_str: str, trained_model_path: str
):
    """Get CLI argument from checkpoint if not passed as CLI argument."""
    if getattr(args, cli_arg_str):
        cli_arg = getattr(args, cli_arg_str)
    else:
        params = get_mlflow_parameters_from_ckpt(trained_model_path)
        cli_arg = ast.literal_eval(params[f"cli_args/{cli_arg_str}"])

    return cli_arg


def get_img_directories_from_ckpt(
    args: argparse.Namespace, trained_model_path: str
) -> list[str]:
    """Get image directories from checkpoint if not passed as CLI argument."""
    # Get dataset directories from ckpt if not defined
    dataset_dirs = get_cli_arg_from_ckpt(
        args=args,
        cli_arg_str="dataset_dirs",
        trained_model_path=trained_model_path,
    )

    # Extract image directories
    images_dirs = prep_img_directories(dataset_dirs)

    return images_dirs


def get_annotation_files_from_ckpt(
    args: argparse.Namespace, trained_model_path: str
) -> list[str]:
    """Get annotation files from checkpoint if not passed as CLI argument."""
    # Get path to input annotation files from ckpt if not defined
    input_annotation_files = get_cli_arg_from_ckpt(
        args=args,
        cli_arg_str="annotation_files",
        trained_model_path=trained_model_path,
    )

    # Get dataset dirs from ckpt if not defined
    dataset_dirs = get_cli_arg_from_ckpt(
        args=args,
        cli_arg_str="dataset_dirs",
        trained_model_path=trained_model_path,
    )

    # Extract annotation files
    annotation_files = prep_annotation_files(
        input_annotation_files, dataset_dirs
    )
    return annotation_files


def get_mlflow_experiment_name_from_ckpt(
    args: argparse.Namespace, trained_model_path: str
) -> str:
    """Define MLflow experiment name from the training job.

    Only used if the experiment name is not passed via CLI.
    """
    if args.experiment_name:
        experiment_name = args.experiment_name
    else:
        params = get_mlflow_parameters_from_ckpt(trained_model_path)
        trained_model_expt_name = params["cli_args/experiment_name"]
        experiment_name = trained_model_expt_name + "_evaluation"

    return experiment_name
