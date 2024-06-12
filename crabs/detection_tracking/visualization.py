import csv
import os
from pathlib import Path
from typing import Any, Optional, Tuple

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
    """
    Draw bounding boxes on the image based on detection results.

    Parameters
    ----------
    frame : np.ndarray
        Image with bounding boxes drawn on it.
    top_left : tuple[float, float]
        Tuple containing (x, y) coordinates of the top-left corner of the bounding box.
    bottom_right : tuple[float, float]
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
    detections: Optional[dict[Any, Any]] = None,
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
    coco_list = COCO_INSTANCE_CATEGORY_NAMES
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


def save_images_with_boxes(
    test_dataloader: torch.utils.data.DataLoader,
    trained_model: torch.nn.Module,
    score_threshold: float,
) -> None:
    """
    Save images with bounding boxes drawn around detected objects.

    Parameters
    ----------
    test_dataloader : DataLoader
        DataLoader for the test dataset.
    trained_model : torch.nn.Module
        The trained object detection model.
    score_threshold : float
        Threshold for object detection.

    Returns
    ----------
        None
    """
    device = (
        torch.device("cuda")
        if torch.cuda.is_available()
        else torch.device("cpu")
    )
    trained_model.eval()
    directory = "results"
    os.makedirs(directory, exist_ok=True)
    with torch.no_grad():
        imgs_id = 0
        for imgs, annotations in test_dataloader:
            imgs_id += 1
            imgs = list(img.to(device) for img in imgs)
            detections = trained_model(imgs)

            image_with_boxes = draw_detection(
                imgs, annotations, detections, score_threshold
            )
            cv2.imwrite(f"{directory}/imgs{imgs_id}.jpg", image_with_boxes)


def plot_sample(imgs: list, row_title: Optional[str] = None, **imshow_kwargs):
    """
    Plot a sample (image & annotations) from a dataset.

    Example usage:
    > full_dataset = CrabsCocoDetection([IMAGES_PATH],[ANNOTATIONS_PATH])
    > sample = full_dataset[0]
    > plt.figure()
    > plot_sample([sample])

    From https://github.com/pytorch/vision/blob/main/gallery/transforms/helpers.py
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


def read_metrics_from_csv(
    filename: str,
) -> Tuple[list[int], list[int], list[int], list[int], list[int], list[float]]:
    """
    Read the tracking output metrics from a CSV file.
    To be called by plot_output_histogram.

    Parameters
    ----------
    filename : str
        Name of the CSV file to read.

    Returns
    -------
    tuple:
        Tuple containing lists of true positives, missed detections,
        false positives, number of switches, and total ground truth for each frame.
    """
    true_positives_list = []
    missed_detections_list = []
    false_positives_list = []
    num_switches_list = []
    total_ground_truth_list = []
    mota_value_list = []

    with open(filename, mode="r") as file:
        reader = csv.DictReader(file)
        for row in reader:
            true_positives_list.append(int(row["True Positives"]))
            missed_detections_list.append(int(row["Missed Detections"]))
            false_positives_list.append(int(row["False Positives"]))
            num_switches_list.append(int(row["Number of Switches"]))
            total_ground_truth_list.append(int(row["Total Ground Truth"]))
            mota_value_list.append(float(row["Mota"]))

    return (
        true_positives_list,
        missed_detections_list,
        false_positives_list,
        num_switches_list,
        total_ground_truth_list,
        mota_value_list,
    )


def plot_output_histogram(filename: str) -> None:
    """
    Plot metrics along with the total ground truth for each frame.

    Example usage:
    > filename = <video_name>_<model_name>_tracking_output.csv
    > plot_output_histogram(filename)

    Parameters
    ----------
    true_positives_list : list[int]
        List of counts of true positives for each frame.
    missed_detections_list : list[int]
        List of counts of missed detections for each frame.
    false_positives_list : list[int]
        List of counts of false positives for each frame.
    num_switches_list : list[int]
        List of counts of identity switches for each frame.
    total_ground_truth_list : list[int]
        List of total ground truth objects for each frame.
    """
    (
        true_positives_list,
        missed_detections_list,
        false_positives_list,
        num_switches_list,
        total_ground_truth_list,
        mota_value_list,
    ) = read_metrics_from_csv(filename)
    filepath = Path(filename)
    plot_name = filepath.name

    num_frames = len(true_positives_list)
    frames = range(1, num_frames + 1)

    plt.figure(figsize=(10, 6))

    overall_mota = sum(mota_value_list) / len(mota_value_list)

    # Calculate percentages
    true_positives_percentage = [
        tp / gt * 100 if gt > 0 else 0
        for tp, gt in zip(true_positives_list, total_ground_truth_list)
    ]
    missed_detections_percentage = [
        md / gt * 100 if gt > 0 else 0
        for md, gt in zip(missed_detections_list, total_ground_truth_list)
    ]
    false_positives_percentage = [
        fp / gt * 100 if gt > 0 else 0
        for fp, gt in zip(false_positives_list, total_ground_truth_list)
    ]
    num_switches_percentage = [
        ns / gt * 100 if gt > 0 else 0
        for ns, gt in zip(num_switches_list, total_ground_truth_list)
    ]

    # Plot metrics
    plt.plot(
        frames,
        true_positives_percentage,
        label=f"True Positives ({sum(true_positives_list)})",
        color="g",
    )
    plt.plot(
        frames,
        missed_detections_percentage,
        label=f"Missed Detections ({sum(missed_detections_list)})",
        color="r",
    )
    plt.plot(
        frames,
        false_positives_percentage,
        label=f"False Positives ({sum(false_positives_list)})",
        color="b",
    )
    plt.plot(
        frames,
        num_switches_percentage,
        label=f"Number of Switches ({sum(num_switches_list)})",
        color="y",
    )

    plt.xlabel("Frame Number")
    plt.ylabel("Percentage of Total Ground Truth (%)")
    plt.title(f"{plot_name}_mota:{overall_mota:.2f}")

    plt.legend()
    plt.savefig(f"{plot_name}.pdf")
    plt.show()
