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
    Save the model and embeddings.

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


def drawing_bbox(
    image_with_boxes,
    top_pt,
    left_pt,
    bottom_pt,
    right_pt,
    colour,
    label_text=None,
) -> None:
    """
    Drawing the bounding boxes on the image, based on detection results.

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

    Note
    ----------
    To draw a rectangle in OpenCV:
        Specify the top-left and bottom-right corners of the rectangle.
    """

    cv2.rectangle(
        image_with_boxes,
        (top_pt, left_pt),
        (bottom_pt, right_pt),
        colour,
        thickness=2,
    )
    if label_text:
        cv2.putText(
            image_with_boxes,
            label_text,
            (top_pt, left_pt),
            cv2.FONT_HERSHEY_SIMPLEX,
            1,
            colour,
            thickness=2,
        )


def drawing_detection(
    imgs, annotations=None, detections=None, score_threshold=None
) -> np.ndarray:
    """
    Drawing the results based on the detection.

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
                drawing_bbox(
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
                    drawing_bbox(
                        image_with_boxes,
                        int((pred_boxes[i][0])[0]),
                        int((pred_boxes[i][0])[1]),
                        int((pred_boxes[i][1])[0]),
                        int((pred_boxes[i][1])[1]),
                        (0, 0, 255),
                        label_text,
                    )

    return image_with_boxes
