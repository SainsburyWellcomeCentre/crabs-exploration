import cv2
import numpy as np
import torch
import torchvision
from detection_utils import coco_category
from sort import Sort


def test_tracking(
    test_dataloader: torch.utils.data.DataLoader,
    trained_model,
    score_threshold: float,
    sort_crab: Sort,
) -> None:
    """
    Test object tracking on a dataset using a trained model.

    Args:
        test_dataloader (torch.utils.data.DataLoader): DataLoader for the test dataset.
        trained_model: The trained object detection model.
        score_threshold (float): The confidence threshold for detection scores.
        sort_crab (Sort): An instance of the sorting algorithm used for tracking.

    Returns:
        None
    """
    device = (
        torch.device("cuda")
        if torch.cuda.is_available()
        else torch.device("cpu")
    )

    coco_list = coco_category()

    with torch.no_grad():
        imgs_id = 0
        for imgs, annotations in test_dataloader:
            imgs_id += 1
            imgs = list(img.to(device) for img in imgs)

            detections = trained_model(imgs, annotations)

            for image, label, prediction in zip(imgs, annotations, detections):
                image = image.cpu().numpy().transpose(1, 2, 0)
                image = (image * 255).astype("uint8")
                image_with_boxes = image.copy()

                pred_score = list(prediction["scores"].detach().cpu().numpy())

                target_boxes = [
                    [(i[0], i[1]), (i[2], i[3])]
                    for i in list(
                        label["boxes"].detach().cpu().detach().numpy()
                    )
                ]
                if pred_score:
                    pred_sort = []
                    pred_t = [pred_score.index(x) for x in pred_score][-1]

                    if all(
                        label == 1
                        for label in list(
                            prediction["labels"].detach().cpu().numpy()
                        )
                    ):
                        pred_class = [
                            coco_list[i]
                            for i in list(
                                prediction["labels"].detach().cpu().numpy()
                            )
                        ]
                        pred_boxes = [
                            [(i[0], i[1]), (i[2], i[3])]
                            for i in list(
                                prediction["boxes"]
                                .detach()
                                .cpu()
                                .detach()
                                .numpy()
                            )
                        ]

                        pred_boxes = pred_boxes[: pred_t + 1]
                        pred_class = pred_class[: pred_t + 1]

                        for pred_i in range(len(pred_boxes)):
                            if (pred_class[pred_i]) == "crab" and pred_score[
                                pred_i
                            ] > score_threshold:
                                bbox = np.asarray(pred_boxes[pred_i])
                                score = np.asarray(pred_score[pred_i])
                                pred_x = np.append(bbox, score)
                                pred_sort.append(pred_x)

                    pred_sort = np.asarray(pred_sort)
                else:
                    pred_sort = np.empty((0, 5))

                sort_bbs_ids = sort_crab.update(pred_sort)

                for sort_i in range(sort_bbs_ids.shape[0]):
                    [x1, y1, x2, y2] = sort_bbs_ids[sort_i, 0:4]
                    cv2.rectangle(
                        image_with_boxes,
                        (int(x1), int(y1)),
                        (int(x2), int(y2)),
                        (0, 0, 255),
                        2,
                    )
                    id_label = f"id : {sort_bbs_ids[sort_i][4]}"
                    cv2.putText(
                        image_with_boxes,
                        id_label,
                        (int(x1), int(y1)),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        1,
                        (0, 0, 255),
                        thickness=2,
                    )

                for target_i in range(len(target_boxes)):
                    cv2.rectangle(
                        image_with_boxes,
                        (
                            int((target_boxes[target_i][0])[0]),
                            int((target_boxes[target_i][0])[1]),
                        ),
                        (
                            int((target_boxes[target_i][1])[0]),
                            int((target_boxes[target_i][1])[1]),
                        ),
                        (0, 255, 0),
                        2,
                    )
                cv2.imwrite(f"imgs{imgs_id}.jpg", image_with_boxes)


def test_detection(
    test_dataloader: torch.utils.data.DataLoader,
    trained_model,
    score_threshold: float,
) -> None:
    """
    Test object detection on a dataset using a trained model.

    Args:
        test_dataloader (torch.utils.data.DataLoader): DataLoader for the test dataset.
        trained_model: The trained object detection model.
        score_threshold (float): The confidence threshold for detection scores.

    Returns:
        None
    """

    device = (
        torch.device("cuda")
        if torch.cuda.is_available()
        else torch.device("cpu")
    )

    total_correct_boxes = 0
    total_gt_boxes = 0

    coco_list = coco_category()

    with torch.no_grad():
        imgs_id = 0
        for imgs, annotations in test_dataloader:
            # print(imgs)
            imgs_id += 1
            imgs = list(img.to(device) for img in imgs)
            targets = [
                {k: v.to(device) for k, v in t.items()} for t in annotations
            ]

            # detections = trained_model(imgs, annotations)
            detections = trained_model(imgs)

            for image, label, prediction in zip(imgs, annotations, detections):
                image = image.cpu().numpy().transpose(1, 2, 0)
                image = (image * 255).astype("uint8")
                image_with_boxes = image.copy()

                pred_score = list(prediction["scores"].detach().cpu().numpy())
                target_boxes = [
                    [(i[0], i[1]), (i[2], i[3])]
                    for i in list(
                        label["boxes"].detach().cpu().detach().numpy()
                    )
                ]
                if pred_score:
                    pred_t = [pred_score.index(x) for x in pred_score][-1]

                    if all(
                        label == 1
                        for label in list(
                            prediction["labels"].detach().cpu().numpy()
                        )
                    ):
                        pred_class = [
                            coco_list[i]
                            for i in list(
                                prediction["labels"].detach().cpu().numpy()
                            )
                        ]
                        pred_boxes = [
                            [(i[0], i[1]), (i[2], i[3])]
                            for i in list(
                                prediction["boxes"]
                                .detach()
                                .cpu()
                                .detach()
                                .numpy()
                            )
                        ]

                        pred_boxes = pred_boxes[: pred_t + 1]
                        pred_class = pred_class[: pred_t + 1]
                        print(len(pred_boxes))
                        for i in range(len(pred_boxes)):
                            if (pred_class[i]) == "crab" and pred_score[
                                i
                            ] > score_threshold:
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

                                label_text = (
                                    f"{pred_class[i]}: {pred_score[i]:.2f}"
                                )
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

                        print(len(target_boxes))
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

                        cv2.imwrite(
                            f"/result/imgs{imgs_id}.jpg", image_with_boxes
                        )

            for target, detection in zip(targets, detections):
                gt_boxes = target["boxes"]
                pred_boxes = detection["boxes"]

                # compare predicted boxes to ground truth boxes
                ious = torchvision.ops.box_iou(pred_boxes, gt_boxes)
                correct_boxes = (ious > score_threshold).sum().item()
                total_correct_boxes += correct_boxes
                total_gt_boxes += len(gt_boxes)

    average_precision = total_correct_boxes / total_gt_boxes
    print(f"Average Precision: {average_precision:.4f}")
