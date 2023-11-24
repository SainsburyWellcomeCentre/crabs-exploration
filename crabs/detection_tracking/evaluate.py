import logging
from typing import List, Tuple

import cv2
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import torch
import torchvision
from detection_utils import coco_category

# Configure logging
logging.basicConfig(level=logging.INFO)

device = (
    torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
)


def drawing_bbox(imgs, annotations, detections, score_threshold) -> np.ndarray:
    """
    Draw bounding boxes on images based on annotations and detections.

    Args:
        imgs: List of images.
        annotations: Ground truth annotations.
        detections: Detected objects.
        score_threshold (float): The confidence threshold for detection scores.

    Returns:
        np.ndarray: Image(s) with bounding boxes drawn on them.
    """

    coco_list = coco_category()
    for image, label, prediction in zip(imgs, annotations, detections):
        image = image.cpu().numpy().transpose(1, 2, 0)
        image = (image * 255).astype("uint8")
        image_with_boxes = image.copy()

        pred_score = list(prediction["scores"].detach().cpu().numpy())
        target_boxes = [
            [(i[0], i[1]), (i[2], i[3])]
            for i in list(label["boxes"].detach().cpu().detach().numpy())
        ]

        if pred_score:
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
                if pred_score[i] > score_threshold:
                    cv2.rectangle(
                        image_with_boxes,
                        (
                            int((pred_boxes[i][0])[0]),
                            int((pred_boxes[i][0])[1]),
                        ),
                        (
                            int((pred_boxes[i][1])[0]),
                            int((pred_boxes[i][1])[1]),
                        ),
                        (0, 0, 255),
                        2,
                    )

                    label_text = f"{pred_class[i]}: {pred_score[i]:.2f}"
                    cv2.putText(
                        image_with_boxes,
                        label_text,
                        (
                            int((pred_boxes[i][0])[0]),
                            int((pred_boxes[i][0])[1]),
                        ),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        1,
                        (0, 0, 255),
                        thickness=2,
                    )
            for i in range(len(target_boxes)):
                cv2.rectangle(
                    image_with_boxes,
                    (
                        int((target_boxes[i][0])[0]),
                        int((target_boxes[i][0])[1]),
                    ),
                    (
                        int((target_boxes[i][1])[0]),
                        int((target_boxes[i][1])[1]),
                    ),
                    (0, 255, 0),
                    2,
                )
    return image_with_boxes


def save_images_with_boxes(
    test_dataloader, trained_model, best_threshold
) -> None:
    """
    Save images with bounding boxes drawn around detected objects.

    Args:
        test_dataloader: DataLoader for the test dataset.
        trained_model: The trained object detection model.
        best_threshold: Threshold for object detection.

    Returns:
        None
    """
    with torch.no_grad():
        imgs_id = 0
        for imgs, annotations in test_dataloader:
            imgs_id += 1
            imgs = list(img.to(device) for img in imgs)
            detections = trained_model(imgs)

            image_with_boxes = drawing_bbox(
                imgs, annotations, detections, best_threshold
            )

            cv2.imwrite(f"results/imgs{imgs_id}.jpg", image_with_boxes)


def compute_precision_recall(
    class_stats,
    score_threshold,
    precision_recall,
    best_f1_score,
    best_threshold,
):
    """
    Compute precision, recall, and best F1 score for object detection.

    Args:
        class_stats: Statistics or information about different classes.
        score_threshold: Threshold for detection scores.
        precision_recall: List to store precision-recall pairs.
        best_f1_score: Best F1 score encountered during computation.
        best_threshold: Threshold corresponding to the best F1 score.

    Returns:
        Tuple containing precision-recall pairs and the best threshold.
    """

    for class_name, stats in class_stats.items():
        precision = stats["tp"] / max(stats["tp"] + stats["fp"], 1)
        recall = stats["tp"] / max(stats["tp"] + stats["fn"], 1)
        precision_recall.append((precision, recall))
        f1_score = (
            2 * (precision * recall) / max((precision + recall), 1e-8)
        )  # Avoid division by zero
        if f1_score > best_f1_score:
            best_f1_score = f1_score
            best_threshold = score_threshold

        logging.info(
            f"Threshold: {score_threshold}, Class: {class_name}, "
            f"Precision: {precision:.4f}, Recall: {recall:.4f}, "
            f"F1 Score: {f1_score:.4f}, "
            f"False Positive: {class_stats['crab']['fp']}, "
            f"False Negative: {class_stats['crab']['fn']}"
        )
    return (
        precision_recall,
        best_threshold,
        best_f1_score,
        best_threshold,
    )


def compute_metrics(
    targets, detections, score_threshold, ious_threshold, class_stats
) -> dict:
    """
    Compute metrics for object detection.

    Args:
        targets: Ground truth annotations.
        detections: Detected objects.
        score_threshold (float): The confidence threshold for detection scores.
        ious_threshold (float): The confidence threshold for IOU.
        class_stats: Statistics or information about different classes.

    Returns:
        dict: Updated class statistics after computation.
    """

    for target, detection in zip(targets, detections):
        gt_boxes = target["boxes"]
        pred_boxes = detection["boxes"]
        pred_labels = detection["labels"]
        pred_scores = detection["scores"]

        bool_valid_indices = pred_scores > score_threshold

        class_stats["crab"]["fp"] += int(
            len(pred_scores) - sum(bool_valid_indices)
        )  # n of detections with score below threshold

        pred_boxes = pred_boxes[bool_valid_indices]
        pred_labels = pred_labels[bool_valid_indices]

        ious = torchvision.ops.box_iou(pred_boxes, gt_boxes)

        max_ious, max_indices = ious.max(dim=1)

        # Identify true positives, false positives, and false negatives
        for idx, iou in enumerate(max_ious):
            if iou.item() > ious_threshold:
                pred_class_idx = pred_labels[idx].item()
                true_label = target["labels"][max_indices[idx]].item()

                if pred_class_idx == true_label == 1:
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

    return class_stats


def test_detection(
    test_dataloader: torch.utils.data.DataLoader,
    trained_model,
    ious_threshold: float,
) -> None:
    """
    Test object detection on a dataset using a trained model.

    Args:
        test_dataloader (torch.utils.data.DataLoader): DataLoader for the test dataset.
        trained_model: The trained object detection model.
        ious_threshold (float): The confidence threshold for IOU.

    Returns:
        None
    """
    score_thresholds = [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
    best_f1_score = 0.0
    best_threshold = 0.0
    precision_recall: List[Tuple[float, float]] = []

    with torch.no_grad():
        all_detections = []
        all_targets = []
        for imgs, annotations in test_dataloader:
            imgs = list(img.to(device) for img in imgs)
            targets = [
                {k: v.to(device) for k, v in t.items()} for t in annotations
            ]
            detections = trained_model(imgs)

            all_detections.extend(detections)
            all_targets.extend(targets)

        for score_threshold in score_thresholds:
            class_stats = {"crab": {"tp": 0, "fp": 0, "fn": 0}}
            class_stats = compute_metrics(
                all_targets,  # one elem per image
                all_detections,
                score_threshold,
                ious_threshold,
                class_stats,
            )

            # Calculate precision, recall, and F1 score for each threshold
            (
                precision_recall,
                best_threshold,
                best_f1_score,
                best_threshold,
            ) = compute_precision_recall(
                class_stats,
                score_threshold,
                precision_recall,
                best_f1_score,
                best_threshold,
            )

        print(best_threshold)
        save_images_with_boxes(test_dataloader, trained_model, best_threshold)
        precisions, recalls = zip(*precision_recall)
        average_precision = sum(precisions) / len(precisions)

        print(
            f"Average Precision (Area Under Precision-Recall Curve): {average_precision:.4f}"
        )

        data = pd.DataFrame({"Recall": recalls, "Precision": precisions})

        # Plotting precision-recall curve with Seaborn style
        plt.figure(figsize=(8, 6))
        sns.lineplot(
            x="Recall", y="Precision", data=data, marker="o", linestyle="-"
        )
        plt.xlabel("Recall")
        plt.ylabel("Precision")
        plt.title("Precision-Recall Curve")

        sns.set(style="dark")
        plt.show()
