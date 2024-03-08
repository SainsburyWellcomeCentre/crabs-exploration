import datetime
import os

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


def draw_bbox(
    frame,
    top_pt,
    left_pt,
    bottom_pt,
    right_pt,
    colour,
    label_text=None,
) -> None:
    """
    Draw the bounding boxes on the image, based on detection results.
    To draw a rectangle in OpenCV:
        Specify the top-left and bottom-right corners of the rectangle.

    Parameters
    ----------
    image_with_boxes : np.ndarray
        Image with bounding boxes drawn on it.
    top_pt : tuple
        Coordinates of the top-left corner of the bounding box.
    left_pt : tuple
        Coordinates of the top-left corner of the bounding box.
    bottom_pt : tuple
        Coordinates of the bottom-right corner of the bounding box.
    right_pt : tuple
        Coordinates of the bottom-right corner of the bounding box.
    colour : tuple
        Color of the bounding box in BGR format.
    label_text : str, optional
        Text to display alongside the bounding box, indicating class and score.

    Returns
    ----------
    None
    """
    # Draw bounding box
    cv2.rectangle(
        frame,
        (top_pt, left_pt),
        (bottom_pt, right_pt),
        colour,
        thickness=2,
    )

    # Add label text if provided
    if label_text:
        cv2.putText(
            frame,
            label_text,
            (top_pt, left_pt),
            cv2.FONT_HERSHEY_SIMPLEX,
            1,
            colour,
            1,
            cv2.LINE_AA,
        )


def draw_detection(
    imgs, annotations=None, detections=None, score_threshold=None
) -> np.ndarray:
    """
    Draw the results based on the detection.

    Parameters
    ----------
    imgs : list
        List of images.
    annotations : dict, optional
        Ground truth annotations.
    detections : dict, optional
        Detected objects.
    score_threshold : float, optional
        The confidence threshold for detection scores.

    Returns
    ----------
    np.ndarray
        Image(s) with bounding boxes drawn on them.
    """

    coco_list = coco_category()
    image_with_boxes = None

    for image, label, prediction in zip(
        imgs, annotations or [], detections or []
    ):
        image = image.cpu().numpy().transpose(1, 2, 0)
        image = (image * 255).astype("uint8")
        image_with_boxes = image.copy()

        if label:
            target_boxes = [
                [(i[0], i[1]), (i[2], i[3])]
                for i in list(label["boxes"].detach().cpu().numpy())
            ]

            for i in range(len(target_boxes)):
                draw_bbox(
                    image_with_boxes,
                    int((target_boxes[i][0])[0]),
                    int((target_boxes[i][0])[1]),
                    int((target_boxes[i][1])[0]),
                    int((target_boxes[i][1])[1]),
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
                        int((pred_boxes[i][0])[0]),
                        int((pred_boxes[i][0])[1]),
                        int((pred_boxes[i][1])[0]),
                        int((pred_boxes[i][1])[1]),
                        (0, 0, 255),
                        label_text,
                    )
    return image_with_boxes


def calculate_iou(box1, box2) -> float:
    """
    Calculate IoU (Intersection over Union) of two bounding boxes.

    Parameters:
    box1 (list): Coordinates [x1, y1, x2, y2] of the first bounding box.
    box2 (list): Coordinates [x1, y1, x2, y2] of the second bounding box.

    Returns:
    float: IoU value.
    """
    x1_box1, y1_box1, x2_box1, y2_box1 = box1
    x1_box2, y1_box2, x2_box2, y2_box2 = box2

    # Calculate intersection coordinates
    x1_intersect = max(x1_box1, x1_box2)
    y1_intersect = max(y1_box1, y1_box2)
    x2_intersect = min(x2_box1, x2_box2)
    y2_intersect = min(y2_box1, y2_box2)

    # Calculate area of intersection rectangle
    intersect_width = max(0, x2_intersect - x1_intersect + 1)
    intersect_height = max(0, y2_intersect - y1_intersect + 1)
    intersect_area = intersect_width * intersect_height

    # Calculate area of individual bounding boxes
    box1_area = (x2_box1 - x1_box1 + 1) * (y2_box1 - y1_box1 + 1)
    box2_area = (x2_box2 - x1_box2 + 1) * (y2_box2 - y1_box2 + 1)

    iou = intersect_area / float(box1_area + box2_area - intersect_area)

    return iou


def draw_gt_tracking(
    gt_boxes: np.ndarray,
    tracked_boxes: np.ndarray,
    frame_number: int,
    iou_threshold: float,
    frame_copy: np.ndarray,
) -> None:
    """
    Track ground truth objects in the frame and draw bounding boxes.

    Parameters
    ----------
    gt_boxes : np.ndarray
        An array containing ground truth bounding boxes of objects for the current frame.
    tracked_boxes : np.ndarray
        An array containing sorted bounding boxes of detected objects.
    frame_number : int
        The frame number to track.
    iou_threshold : float
        The intersection over union threshold for considering a match.
    frame_copy : np.ndarray
        A copy of the input frame for drawing bounding boxes.
    """

    for gt_box in gt_boxes:
        x_gt, y_gt, x2_gt, y2_gt, gt_id = gt_box

        for tracked_box in tracked_boxes:
            x1_track, y1_track, x2_track, y2_track, track_id = tracked_box
            iou = calculate_iou(
                [x_gt, y_gt, x2_gt, y2_gt],
                [x1_track, y1_track, x2_track, y2_track],
            )
            x_gt, y_gt, x2_gt, y2_gt = map(int, [x_gt, y_gt, x2_gt, y2_gt])
            x1_track, y1_track, x2_track, y2_track = map(
                int, [x1_track, y1_track, x2_track, y2_track]
            )

            if iou > iou_threshold:
                draw_bbox(
                    frame_copy,
                    x_gt,
                    y_gt,
                    x2_gt,
                    y2_gt,
                    (0, 255, 0),
                    f"gt id : {int(gt_id)}",
                )

                draw_bbox(
                    frame_copy,
                    x1_track,
                    y1_track,
                    x2_track,
                    y2_track,
                    (0, 0, 255),
                    f"track id : {int(track_id)}",
                )

    return frame_copy
