import datetime
import os
from typing import Any, Dict, Optional, Tuple

import cv2
import numpy as np
import torch


def coco_category():
    """
    Get the COCO instance category names.

    Returns
    -------
    list of str
        List of COCO instance category names.
    """
    COCO_INSTANCE_CATEGORY_NAMES = [
        "__background__",
        "crab",
    ]
    return COCO_INSTANCE_CATEGORY_NAMES


def save_model(model: torch.nn.Module):
    """
    Save the trained model.

    Parameters
    ----------
    model : torch.nn.Module
        The PyTorch model to be saved.

    Returns
    -------
    None

    Notes
    -----
    This function saves the provided PyTorch model to a file with a unique
    filename based on the current date and time. The filename format is
    'model_<timestamp>.pt'.

    """
    current_time = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    directory = "model"
    os.makedirs(directory, exist_ok=True)
    filename = f"{directory}/model_{current_time}.pt"

    print(filename)
    torch.save(model, filename)
    print("Model Saved")

    return filename


def draw_bbox(
    frame: np.ndarray,
    top_left: Tuple[float, float],
    bottom_right: Tuple[float, float],
    colour: tuple,
    label_text: Optional[str] = None,
) -> None:
    """
    Draw bounding boxes on the image based on detection results.

    Parameters
    ----------
    frame : np.ndarray
        Image with bounding boxes drawn on it.
    top_left : Tuple[int, int]
        Tuple containing (x, y) coordinates of the top-left corner of the bounding box.
    bottom_right : Tuple[int, int]
        Tuple containing (x, y) coordinates of the bottom-right corner of the bounding box.
    colour : tuple
        Color of the bounding box in BGR format.
    label_text : str, optional
        Text to display alongside the bounding box, indicating class and score.

    Returns
    -------
    None
    """
    # Draw bounding box
    cv2.rectangle(
        frame,
        (int(top_left[0]), int(top_left[1])),
        (int(bottom_right[0]), int(bottom_right[1])),
        colour,
        thickness=2,
    )

    # Add label text if provided
    if label_text:
        cv2.putText(
            frame,
            label_text,
            (int(top_left[0]), int(top_left[1])),
            cv2.FONT_HERSHEY_SIMPLEX,
            1,
            colour,
            2,
            cv2.LINE_AA,
        )


def draw_detection(
    imgs: list,
    annotations: dict,
    detections: Optional[Dict[Any, Any]] = None,
    score_threshold: Optional[float] = None,
) -> np.ndarray:
    """
    Draw the results based on the detection.

    Parameters
    ----------
    imgs : list
        List of images.
    annotations : dict
        Ground truth annotations.
    detections : dict, optional
        Detected objects.
    score_threshold : float, optional
        The confidence threshold for detection scores.

    Returns
    -------
    np.ndarray
        Image(s) with bounding boxes drawn on them.
    """
    coco_list = coco_category()
    image_with_boxes = None

    for image, label, prediction in zip(
        imgs, annotations, detections or [None] * len(imgs)
    ):
        image = image.cpu().numpy().transpose(1, 2, 0)
        image = (image * 255).astype("uint8")
        image_with_boxes = image.copy()

        target_boxes = [
            [(i[0], i[1]), (i[2], i[3])]
            for i in list(label["boxes"].detach().cpu().numpy())
        ]

        for i in range(len(target_boxes)):
            draw_bbox(
                image_with_boxes,
                ((target_boxes[i][0])[0], (target_boxes[i][0])[1]),
                ((target_boxes[i][1])[0], (target_boxes[i][1])[1]),
                colour=(0, 255, 0),
            )
        if prediction:
            pred_score = list(prediction["scores"].detach().cpu().numpy())
            pred_t = [pred_score.index(x) for x in pred_score][-1]

            pred_class = [
                coco_list[i]
                for i in list(prediction["labels"].detach().cpu().numpy())
            ]

            pred_boxes = [
                [(i[0], i[1]), (i[2], i[3])]
                for i in list(
                    prediction["boxes"].detach().cpu().detach().numpy()
                )
            ]

            pred_boxes = pred_boxes[: pred_t + 1]
            pred_class = pred_class[: pred_t + 1]
            for i in range(len(pred_boxes)):
                if pred_score[i] > (score_threshold or 0):
                    label_text = f"{pred_class[i]}: {pred_score[i]:.2f}"
                    draw_bbox(
                        image_with_boxes,
                        (
                            (pred_boxes[i][0])[0],
                            (pred_boxes[i][0])[1],
                        ),
                        (
                            (pred_boxes[i][1])[0],
                            (pred_boxes[i][1])[1],
                        ),
                        (0, 0, 255),
                        label_text,
                    )
    return image_with_boxes
