import logging

import cv2
import numpy as np
import torch
import torchvision
from detection_utils import coco_category

# Configure logging
logging.basicConfig(level=logging.INFO)


def drawing_bbox(imgs, annotations, detections, score_threshold) -> np.ndarray:
    """
    Draw bounding boxes on images based on annotations and detections.

    Parameters
    ----------
    imgs : list
        List of images.
    annotations : dict
        Ground truth annotations.
    detections : dict
        Detected objects.
    score_threshold ; float
        The confidence threshold for detection scores.

    Returns
    ----------
    np.ndarray
        Image(s) with bounding boxes drawn on them.
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

            image_with_boxes = drawing_bbox(
                imgs, annotations, detections, score_threshold
            )

            cv2.imwrite(f"results/imgs{imgs_id}.jpg", image_with_boxes)


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

        logging.info(
            f"Precision: {precision:.4f}, Recall: {recall:.4f}, "
            f"False Positive: {class_stats['crab']['fp']}, "
            f"False Negative: {class_stats['crab']['fn']}"
        )


def compute_confusion_metrics(
    targets, detections, ious_threshold, class_stats
) -> dict:
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
    dict
        Updated class statistics after computation.
    """

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

    return class_stats


def evaluate_detection(
    test_dataloader: torch.utils.data.DataLoader,
    trained_model,
    ious_threshold: float,
    score_threshold: float,
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
    device = (
        torch.device("cuda")
        if torch.cuda.is_available()
        else torch.device("cpu")
    )
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

        class_stats = {"crab": {"tp": 0, "fp": 0, "fn": 0}}
        class_stats = compute_confusion_metrics(
            all_targets,  # one elem per image
            all_detections,
            ious_threshold,
            class_stats,
        )

        print(type(class_stats))

        # Calculate precision, recall, and F1 score for each threshold
        compute_precision_recall(class_stats)

        save_images_with_boxes(
            test_dataloader, trained_model, score_threshold, device
        )
