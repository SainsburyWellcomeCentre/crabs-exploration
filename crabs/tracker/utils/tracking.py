"""Utility functions for tracking."""

import json
from pathlib import Path
from typing import Any

import pandas as pd
import torch


def format_bbox_predictions_for_sort(
    prediction_dict: dict, score_threshold: float
) -> torch.Tensor:
    """Put predictions in format expected by SORT.

    Parameters
    ----------
    prediction_dict : dict
        Dictionary containing predicted bounding boxes, scores,
        and labels.

    score_threshold : float
        The threshold score for filtering out low-confidence predictions.

    Returns
    -------
    torch.Tensor:
        A torch tensor containing bounding boxes and scores along
        columns.

    """
    # Format as a tensor with scores as last column
    predictions_tensor = torch.hstack(
        (
            prediction_dict["boxes"],
            prediction_dict["scores"].unsqueeze(dim=1),
        )
    )

    # Filter rows in tensor based on last column
    # if pred_score > score_threshold:
    return predictions_tensor[predictions_tensor[:, -1] > score_threshold]


def extract_bounding_box_info(row: list[str]) -> dict[str, Any]:
    """Extract bounding box information from a row of data.

    Parameters
    ----------
    row : list[str]
        A list representing a row of data containing information about a
        bounding box.

    Returns
    -------
    dict[str, Any]:
        A dictionary containing the extracted bounding box information.

    """
    filename = row[0]
    region_shape_attributes = json.loads(row[5])
    region_attributes = json.loads(row[6])

    x = region_shape_attributes["x"]
    y = region_shape_attributes["y"]
    width = region_shape_attributes["width"]
    height = region_shape_attributes["height"]
    track_id = region_attributes["track"]

    frame_number = int(filename.split("_")[-1].split(".")[0])
    return {
        "frame_number": frame_number,
        "x": x,
        "y": y,
        "width": width,
        "height": height,
        "id": track_id,
    }


def save_tracking_mota_metrics(
    tracking_output_dir: Path,
    track_results: dict[str, Any],
) -> None:
    """Save tracking metrics to a CSV file."""
    track_df = pd.DataFrame(track_results)
    output_filename = f"{tracking_output_dir}/tracking_metrics_output.csv"
    track_df.to_csv(output_filename, index=False)
