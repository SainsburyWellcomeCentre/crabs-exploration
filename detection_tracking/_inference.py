import numpy as np
import cv2
from _utils import coco_category
from sort import Sort


def inference_tracking(
    frame: np.ndarray,
    prediction: list,
    pred_score: list,
    score_threshold: float,
    sort_crab: Sort,
) -> np.ndarray:
    """
    Perform object tracking inference on a frame using prediction results.

    Args:
        frame (np.ndarray): The input frame as a NumPy array.
        prediction (list): List containing prediction results.
        pred_score (list): List of prediction scores.
        score_threshold (float): The confidence threshold for detection scores.
        sort_crab (Sort): An instance of the sorting algorithm used for tracking.

    Returns:
        np.ndarray: The input frame with tracking bounding boxes and IDs drawn on it.
    """

    coco_list = coco_category()

    if pred_score:
        pred_sort = []
        pred_t = [pred_score.index(x) for x in pred_score][-1]

        if all(
            label == 1 for label in list(prediction[0]["labels"].detach().cpu().numpy())
        ):
            pred_class = [
                coco_list[i]
                for i in list(prediction[0]["labels"].detach().cpu().numpy())
            ]
            pred_boxes = [
                [(i[0], i[1]), (i[2], i[3])]
                for i in list(prediction[0]["boxes"].detach().cpu().detach().numpy())
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
            if pred_sort:
                pred_sort = np.asarray(pred_sort)
            else:
                pred_sort = np.empty((0, 5))
        else:
            pred_sort = np.empty((0, 5))
    else:
        pred_sort = np.empty((0, 5))

    sort_bbs_ids = sort_crab.update(pred_sort)

    for sort_i in range(sort_bbs_ids.shape[0]):
        [x1, y1, x2, y2] = sort_bbs_ids[sort_i, 0:4]
        cv2.rectangle(
            frame,
            (int(x1), int(y1)),
            (int(x2), int(y2)),
            (0, 0, 255),
            2,
        )
        id_label = f"id : {sort_bbs_ids[sort_i][4]}"
        cv2.putText(
            frame,
            id_label,
            (int(x1), int(y1)),
            cv2.FONT_HERSHEY_SIMPLEX,
            1,
            (0, 0, 255),
            thickness=2,
        )

    return frame


def inference_detection(
    frame: np.ndarray,
    prediction: list,
    pred_score: list,
    score_threshold: float,
) -> np.ndarray:
    """
    Perform object detection inference on a frame using prediction results.

    Args:
        frame (np.ndarray): The input frame as a NumPy array.
        prediction (list): List containing prediction results.
        pred_score (list): List of prediction scores.
        score_threshold (float): The confidence threshold for detection scores.

    Returns:
        np.ndarray: The input frame with tracking bounding boxes and IDs drawn on it.
    """

    coco_list = coco_category()

    if pred_score:
        pred_t = [pred_score.index(x) for x in pred_score][-1]

        if all(
            label == 1 for label in list(prediction[0]["labels"].detach().cpu().numpy())
        ):
            pred_class = [
                coco_list[i]
                for i in list(prediction[0]["labels"].detach().cpu().numpy())
            ]
            pred_boxes = [
                [(i[0], i[1]), (i[2], i[3])]
                for i in list(prediction[0]["boxes"].detach().cpu().detach().numpy())
            ]

            pred_boxes = pred_boxes[: pred_t + 1]
            pred_class = pred_class[: pred_t + 1]

            for i in range(len(pred_boxes)):
                if (pred_class[i]) == "crab" and pred_score[i] > score_threshold:
                    cv2.rectangle(
                        frame,
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
                        frame,
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

    return frame
