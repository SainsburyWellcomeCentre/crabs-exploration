import logging
import os
from typing import Tuple

import cv2
import torch
import torchvision

from crabs.detection_tracking.detection_utils import draw_detection

logging.basicConfig(level=logging.INFO)


def save_images_with_boxes(
    test_dataloader, trained_model, score_threshold, device
) -> None:
    """
    Save images with bounding boxes drawn around detected objects.

    Parameters
    ----------
    test_dataloader :
        DataLoader for the test dataset.
    trained_model :
        The trained object detection model.
    score_threshold : float
        Threshold for object detection.

    Returns
    ----------
        None
    """
    with torch.no_grad():
        imgs_id = 0
        for imgs, annotations in test_dataloader:
            imgs_id += 1
            imgs = list(img.to(device) for img in imgs)
            detections = trained_model(imgs)

            image_with_boxes = draw_detection(
                imgs, annotations, detections, score_threshold
            )
            directory = "results"
            os.makedirs(directory, exist_ok=True)
            cv2.imwrite(f"{directory}/imgs{imgs_id}.jpg", image_with_boxes)


def compute_precision_recall(class_stats):
    """
    Compute precision and recall.

    Parameters
    ----------
    class_stats : dict
        Statistics or information about different classes.

    Returns
    ----------
    None
    """

    for _, stats in class_stats.items():
        precision = stats["tp"] / max(stats["tp"] + stats["fp"], 1)
        recall = stats["tp"] / max(stats["tp"] + stats["fn"], 1)

    return precision, recall


def compute_confusion_matrix_elements(
    targets, detections, ious_threshold
) -> Tuple[float, float, dict]:
    """
    Compute metrics (true positive, false positive, false negative) for object detection.

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
    ----------
    Tuple[float, float]
        precision and recall
    """
    class_stats = {"crab": {"tp": 0, "fp": 0, "fn": 0}}
    print(targets)
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

        for target_box_index, target_box in enumerate(gt_boxes):
            found_match = False
            for idx, iou in enumerate(max_ious):
                if (
                    iou.item()
                    > ious_threshold  # we need this condition because the max overlap is not necessarily above the threshold
                    and max_indices[idx]
                    == target_box_index  # the matching index is the index of the GT box with which it has max overlap
                ):
                    # There's an IoU match and the matched index corresponds to the current target_box_index
                    found_match = True
                    break  # Exit loop, a match was found

            if not found_match:
                # print(found_match)
                class_stats["crab"][
                    "fn"
                ] += 1  # Ground truth box has no corresponding detection

    precision, recall, class_stats = compute_precision_recall(class_stats)

    return precision, recall, class_stats