import os
import numpy as np
import cv2
import datetime
import json
import torch
import torchvision.transforms as transforms

from crabs.detection_tracking.myfaster_rcnn_dataset import myFasterRCNNDataset


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


# collate_fn needs for batch
def collate_fn(batch):
    """
    Collates a batch of samples into a structured format that is expected in Pytorch.

    This function takes a list of samples, where each sample can be any data structure.
    It filters out any `None` values from the batch and then groups the
    remaining samples into a structured format. The structured format
    is a tuple of lists, where each list contains the elements from the
    corresponding position in the samples.

    Parameters
    ----------
    batch : list
        A list of samples, where each sample can be any data structure.

    Returns
    -------
    collated : Optional[Tuple[List[Any]]]
        A tuple of lists, where each list contains the elements from the corresponding
        position in the samples. If the input batch is empty or contains only `None`
        values, the function returns `None`.
    """
    batch = [sample for sample in batch if sample is not None]

    if len(batch) == 0:
        return None

    return tuple(zip(*batch))


def create_dataloader(
    my_dataset: torch.utils.data.Dataset, batch_size: int
) -> torch.utils.data.DataLoader:
    """
    Creates a customized DataLoader for a given dataset.

    This function constructs a DataLoader using the provided dataset and batch size.
    It also applies shuffling to the data, employs multiple worker processes for
    data loading, uses a custom collate function to process batches, and enables
    pinning memory for optimized data transfer to GPU.

    Parameters
    ----------
    my_dataset : torch.utils.data.Dataset
        The dataset containing the samples to be loaded.

    batch_size : int
        The number of samples in each batch.

    Returns
    -------
    data_loader : torch.utils.data.DataLoader
        A DataLoader configured with the specified settings for loading data
        from the provided dataset.

    Notes
    -----
    This function provides a convenient way to create a DataLoader with custom settings
    tailored for specific data loading needs.

    Examples
    --------
    >>> my_dataset = CustomDataset()
    >>> data_loader = create_dataloader(my_dataset, batch_size=32)
    >>> for batch in data_loader:
    ...     # Training loop or batch processing
    """
    data_loader = torch.utils.data.DataLoader(
        my_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=4,
        collate_fn=collate_fn,
    )
    return data_loader


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
        This function does not return anything.

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


def get_train_transform() -> transforms.Compose:
    """
    Get a composed transformation for training data for data augmentation and
    transform the data to tensor.

    Returns
    -------
    transforms.Compose
        A composed transformation that includes the specified operations.

    Notes
    -----
    This function returns a composed transformation for use with training data.
    The following transforms are applied in sequence:
    1. Apply color jittering with brightness adjustment (0.5) and hue (0.3).
    2. Apply Gaussian blur with kernel size of (5, 9) and sigma (0.1, 5.0)
    3. Convert the image to a PyTorch tensor.

    Examples
    --------
    >>> train_transform = get_train_transform()
    >>> dataset = MyDataset(transform=train_transform)
    """
    # TODO: testing with different transforms
    custom_transforms = []
    custom_transforms.append(transforms.ColorJitter(brightness=0.5, hue=0.3))
    custom_transforms.append(
        transforms.GaussianBlur(kernel_size=(5, 9), sigma=(0.1, 5.0))
    )
    custom_transforms.append(transforms.ToTensor())

    return transforms.Compose(custom_transforms)


def get_eval_transform() -> transforms.Compose:
    """
    Get a composed transformation for evaluation data which
    transform the data to tensor.

    Returns
    -------
    transforms.Compose
        A composed transformation that includes the specified operations.
    """
    custom_transforms = []
    custom_transforms.append(transforms.ToTensor())

    return transforms.Compose(custom_transforms)


def load_dataset(main_dir, annotation, batch_size, training=False):
    """Load images and annotation file for training or evaluation"""

    with open(annotation) as json_file:
        coco_data = json.load(json_file)

        file_paths = []
        for image_info in coco_data["images"]:
            image_id = image_info["id"]
            image_id -= 1
            image_file = image_info["file_name"]
            video_file = image_file.split("_")[1]

            if not training and (
                video_file == "09.08.2023-03-Left"
                or video_file == "10.08.2023-01-Left"
                or video_file == "10.08.2023-01-Right"
            ):
                continue

            # For training, take the first 40 frames per video as training data
            # For evaluation, take the remaining frames per video
            if training:
                if image_id % 50 < 40:
                    file_paths.append(image_file)
            else:
                if image_id % 50 >= 40:
                    file_paths.append(image_file)

    dataset = myFasterRCNNDataset(
        main_dir,
        file_paths,
        annotation,
        transforms=get_train_transform() if training else get_eval_transform(),
    )

    dataloader = create_dataloader(dataset, batch_size)

    return dataloader


def drawing_bbox(
    image_with_boxes, top_pt, left_pt, bottom_pt, right_pt, colour, label_text=None
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
