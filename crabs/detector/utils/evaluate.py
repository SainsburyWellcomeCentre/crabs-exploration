"""Utils used in evaluation."""

import argparse
import ast
import logging
import sys
from pathlib import Path

import torchvision
import yaml  # type: ignore

from crabs.detector.utils.detection import (
    prep_annotation_files,
    prep_img_directories,
)

logging.basicConfig(level=logging.INFO)


def compute_precision_recall(class_stats: dict) -> tuple[float, float, dict]:
    """Compute precision and recall.

    Parameters
    ----------
    class_stats : dict
        Statistics or information about different classes.

    Returns
    -------
    Tuple[float, float]
        precision and recall

    """
    for _, stats in class_stats.items():
        precision = stats["tp"] / max(stats["tp"] + stats["fp"], 1)
        recall = stats["tp"] / max(stats["tp"] + stats["fn"], 1)

    return precision, recall, class_stats


def compute_confusion_matrix_elements(
    targets: list, detections: list, ious_threshold: float
) -> tuple[float, float, dict]:
    """Compute detection metrics.

    Compute true positive, false positive, and false negative values.

    Parameters
    ----------
    targets : list
        Ground truth annotations.
    detections : list
        Detected objects.
    ious_threshold  : float
        The threshold value for the intersection-over-union (IOU).
        Only detections whose IOU relative to the ground truth is above the
        threshold are true positive candidates.
    class_stats : dict
        Statistics or information about different classes.

    Returns
    -------
    Tuple[float, float]
        precision and recall

    """
    class_stats = {"crab": {"tp": 0, "fp": 0, "fn": 0}}
    for target, detection in zip(targets, detections):
        gt_boxes = target["boxes"]
        pred_boxes = detection["boxes"]
        pred_labels = detection["labels"]

        ious = torchvision.ops.box_iou(pred_boxes, gt_boxes)

        max_ious, max_indices = ious.max(dim=1)

        # Identify true positives, false positives, and false negatives
        for idx, iou in enumerate(max_ious):
            if iou.item() > ious_threshold:
                pred_class_idx = pred_labels[idx].item()
                true_label = target["labels"][max_indices[idx]].item()

                if pred_class_idx == true_label:
                    class_stats["crab"]["tp"] += 1
                else:
                    class_stats["crab"]["fp"] += 1
            else:
                class_stats["crab"]["fp"] += 1

        for target_box_index, _target_box in enumerate(gt_boxes):
            found_match = False
            for idx, iou in enumerate(max_ious):
                if (
                    iou.item() > ious_threshold
                    # we need this condition because the max overlap
                    # is not necessarily above the threshold
                    and max_indices[idx] == target_box_index
                    # the matching index is the index of the GT
                    # box with which it has max overlap
                ):
                    # There's an IoU match and the matched index corresponds
                    # to the current target_box_index
                    found_match = True
                    break  # Exit loop, a match was found

            if not found_match:
                # print(found_match)
                class_stats["crab"]["fn"] += (
                    1  # Ground truth box has no corresponding detection
                )

    precision, recall, class_stats = compute_precision_recall(class_stats)

    return precision, recall, class_stats


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

    return params


def get_config_from_ckpt(config_file: str, trained_model_path: str) -> dict:
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
    """Get CLI argument from checkpoint if not in args."""
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
