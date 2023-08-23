import os
import datetime

import torch
import torchvision.transforms as transforms
from PIL import Image
from pycocotools.coco import COCO


class myFasterRCNNDataset(torch.utils.data.Dataset):
    """Custom Pytorch dataset class for Faster RCNN object detection
    using COCO-style annotation.

    Args
    ----------
    data_dir : str
        Path to the directory containing image data.
    annotation : str
        Path to the coco-style annotation file.
    transforms : callable, optional
        A function to apply to the images

    Attributes
    ----------
    data_dir : str
        Path to the directory containing image data.
    annotation : str
        Path to the coco-style annotation file.
    ids : list
        List of image IDs from the COCO annotation.
    transforms : callable, optional
        A function to apply to the images

    Returns
    ----------
    tuple : A tuple containing an image tensor and a dictionary of annotations

    """

    def __init__(self, data_dir, annotation, transforms=None):
        self.data_dir = data_dir
        self.coco = COCO(annotation)
        self.transforms = transforms
        self.ids = list(sorted(self.coco.imgs.keys()))

    def __getitem__(self, index):
        """Get the image and associated annotations at the specified index.

        Args
        ----------
        index : str
            Index of the sample to retrieve.

        Returns
        ----------
        tuple: A tuple containing the image tensor and a dictionary of annotations.

        Note
        ----------
        The annotations dictionary contains the following keys:
            - 'image': The image tensor.
            - 'annotations': A dictionary containing object annotations with keys:
            - 'boxes': Bounding box coordinates (xmin, ymin, xmax, ymax).
            - 'labels': Class labels for each object.
            - 'image_id': Image ID.
            - 'area': Area of each object.
        -    'iscrowd': Flag indicating whether the object is a crowd.
        """

        img_id = self.ids[index]
        ann_ids = self.coco.getAnnIds(imgIds=img_id)
        coco_annotation = self.coco.loadAnns(ann_ids)

        # If there are no annotations, skip this frame (unlabeled frame)
        if not coco_annotation:
            return None

        img_info = self.coco.loadImgs(ids=img_id)[0]
        img_path = os.path.join(self.data_dir, img_info["file_name"])
        img = Image.open(img_path).convert("RGB")

        num_objs = len(coco_annotation)

        # Bounding boxes for objects
        # In coco format, bbox = [xmin, ymin, width, height]
        # In pytorch, the input should be [xmin, ymin, xmax, ymax]
        boxes = []
        for i in range(num_objs):
            xmin = coco_annotation[i]["bbox"][0]
            ymin = coco_annotation[i]["bbox"][1]
            xmax = xmin + coco_annotation[i]["bbox"][2]
            ymax = ymin + coco_annotation[i]["bbox"][3]
            boxes.append([xmin, ymin, xmax, ymax])
        boxes = torch.as_tensor(boxes, dtype=torch.float32)
        labels = torch.ones((num_objs,), dtype=torch.int64)
        img_id = torch.tensor([img_id])

        areas = []
        for i in range(num_objs):
            areas.append(coco_annotation[i]["area"])
        areas = torch.as_tensor(areas, dtype=torch.float32)
        iscrowd = torch.zeros((num_objs,), dtype=torch.int64)

        # Annotation is in dictionary format
        my_annotation = {}
        my_annotation["boxes"] = boxes
        my_annotation["labels"] = labels
        my_annotation["image_id"] = img_id
        my_annotation["area"] = areas
        my_annotation["iscrowd"] = iscrowd

        if self.transforms is not None:
            img = self.transforms(img)

        return img, my_annotation

    def __len__(self):
        """Get the total number of samples in the dataset.

        Returns
        ----------
            int: The number of samples in the dataset
        """
        return len(self.ids)


def coco_category():
    COCO_INSTANCE_CATEGORY_NAMES = [
        "__background__",
        "crab",
    ]
    return COCO_INSTANCE_CATEGORY_NAMES


def collate_fn(batch):
    """
    Collates a batch of samples into a structured format.

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

    Notes
    -----
    This function is useful for collating samples before passing them into a data loader
    or a batching process.

    Examples
    --------
    >>> sample1 = (1, 'a')
    >>> sample2 = (2, 'b')
    >>> sample3 = None
    >>> collate_fn([sample1, sample2, sample3])
    ((1, 2), ('a', 'b'))
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
        pin_memory=True,
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
    filename = f"model_{current_time}.pt"
    torch.save(model, filename)
    print("Model Saved")


def get_train_transform() -> transforms.Compose:
    """
    Get a composed transformation for training data.

    Returns
    -------
    transforms.Compose
        A composed transformation that includes the specified operations.

    Notes
    -----
    This function returns a composed transformation for use with training data.
    The following transforms are applied in sequence:
    2. Apply color jittering with brightness adjustment (0.5) and hue (0.3).
    3. Convert the image to a PyTorch tensor.

    Examples
    --------
    >>> train_transform = get_train_transform()
    >>> dataset = MyDataset(transform=train_transform)
    """
    # TODO: testing with different transforms
    custom_transforms = [
        transforms.ColorJitter(brightness=0.5, hue=0.3),
        transforms.ToTensor(),
    ]

    return transforms.Compose(custom_transforms)


def get_test_transform() -> transforms.Compose:
    """
    Get a composed transformation for test data.

    Returns
    -------
    transforms.Compose
        A composed transformation that includes the specified operations.

    Examples
    --------
    >>> test_transform = get_test_transform()
    >>> dataset = MyDataset(transform=test_transform)
    """
    custom_transforms = []
    custom_transforms.append(transforms.ToTensor())

    return transforms.Compose(custom_transforms)
