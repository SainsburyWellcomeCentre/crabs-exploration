"""Utilities for visualizing object detection results."""

import os
from datetime import datetime
from typing import Any, Optional

import cv2
import matplotlib.pyplot as plt
import numpy as np
import torch
from torchvision import tv_tensors
from torchvision.transforms.v2 import functional as F
from torchvision.utils import draw_bounding_boxes, draw_segmentation_masks

COCO_INSTANCE_CATEGORY_NAMES = [
    "__background__",
    "crab",
]


def draw_bbox(
    frame: np.ndarray,
    top_left: tuple[float, float],
    bottom_right: tuple[float, float],
    colour: tuple,
    label_text: Optional[str] = None,
) -> None:
    """Draw bounding boxes on the image based on detection results.

    Parameters
    ----------
    frame : np.ndarray
        Image with bounding boxes drawn on it.
    top_left : tuple[float, float]
        Tuple containing (x, y) coordinates of the top-left corner of the
        bounding box.
    bottom_right : tuple[float, float]
        Tuple containing (x, y) coordinates of the bottom-right corner of the
        bounding box.
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


def draw_detections(
    imgs: list,
    annotations: dict,
    detections: Optional[dict[Any, Any]] = None,
    score_threshold: Optional[float] = None,
) -> list[np.ndarray]:
    """Draw the results based on the detection.

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
    list[np.ndarray]
        Image(s) with bounding boxes drawn on them.

    """
    coco_list = COCO_INSTANCE_CATEGORY_NAMES

    list_images_with_boxes = []
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
        list_images_with_boxes.append(image_with_boxes)

    return list_images_with_boxes


def save_images_with_boxes(
    dataloader: torch.utils.data.DataLoader,
    trained_model: torch.nn.Module,
    output_dir: str,
    score_threshold: float,
) -> None:
    """Save images with bounding boxes drawn around detected objects.

    Parameters
    ----------
    dataloader : DataLoader
        DataLoader with the images to save.
    trained_model : torch.nn.Module
        The trained object detection model.
    output_dir : str
        Path to directory to save the images with bounding boxes.
        The directory name will be added a timestamp.
    score_threshold : float
        Threshold for object detection.

    Returns
    -------
        None

    """
    device = (
        torch.device("cuda")
        if torch.cuda.is_available()
        else torch.device("cpu")
    )

    trained_model.to(device)
    trained_model.eval()

    # set output directory
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = f"{output_dir}_{timestamp}"
    os.makedirs(output_dir, exist_ok=True)

    with torch.no_grad():
        for batch_id, (imgs, annotations) in enumerate(dataloader):
            imgs = list(img.to(device) for img in imgs)

            detections = trained_model(imgs)

            imgs_with_boxes = draw_detections(
                imgs, annotations, detections, score_threshold
            )

            for img_id, img_with_boxes in enumerate(imgs_with_boxes):
                cv2.imwrite(
                    f"{output_dir}/img_batch_{batch_id}_{img_id}.jpg",
                    img_with_boxes,
                )


def plot_sample(  # noqa: C901
    imgs: list, row_title: Optional[str] = None, **imshow_kwargs
):
    """Plot a sample (image & annotations) from a dataset.

    Example usage:
    > full_dataset = CrabsCocoDetection([IMAGES_PATH],[ANNOTATIONS_PATH])
    > sample = full_dataset[0]
    > plt.figure()
    > plot_sample([sample])

    From:
    https://github.com/pytorch/vision/blob/main/gallery/transforms/helpers.py
    """
    if not isinstance(imgs[0], list):
        # Make a 2d grid even if there's just 1 row
        imgs = [imgs]

    num_rows = len(imgs)
    num_cols = len(imgs[0])
    _, axs = plt.subplots(nrows=num_rows, ncols=num_cols, squeeze=False)
    for row_idx, row in enumerate(imgs):
        for col_idx, img in enumerate(row):
            boxes = None
            masks = None
            if isinstance(img, tuple):
                img, target = img
                if isinstance(target, dict):
                    boxes = target.get("boxes")
                    masks = target.get("masks")
                elif isinstance(target, tv_tensors.BoundingBoxes):
                    boxes = target
                else:
                    raise ValueError(f"Unexpected target type: {type(target)}")
            img = F.to_image(img)
            if img.dtype.is_floating_point and img.min() < 0:
                # Poor man's re-normalization for the colors to be OK-ish. This
                # is useful for images coming out of Normalize()
                img -= img.min()
                img /= img.max()

            img = F.to_dtype(img, torch.uint8, scale=True)
            if boxes is not None:
                img = draw_bounding_boxes(img, boxes, colors="yellow", width=3)
            if masks is not None:
                img = draw_segmentation_masks(
                    img,
                    masks.to(torch.bool),
                    colors=["green"] * masks.shape[0],
                    alpha=0.65,
                )

            ax = axs[row_idx, col_idx]
            ax.imshow(img.permute(1, 2, 0).numpy(), **imshow_kwargs)
            ax.set(xticklabels=[], yticklabels=[], xticks=[], yticks=[])

    if row_title is not None:
        for row_idx in range(num_rows):
            axs[row_idx, 0].set(ylabel=row_title[row_idx])

    plt.tight_layout()
