import json

import lightning as pl
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader
from torchvision.transforms import transforms

from crabs.detection_tracking.myfaster_rcnn_dataset import myFasterRCNNDataset


def collate_fn(batch):
    """
    Collates a batch of samples into a structured format that is expected in Pytorch.

    This function takes a list of samples, where each sample can be any data structure.
    It filters out any `None` values from the batch, and then groups the
    remaining samples into a tuple of lists, where each list contains the elements from the
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


class myDataModule(pl.LightningDataModule):
    """
    Data module for handling data loading and preprocessing for object detection models.

    Parameters:
    -----------
    main_dir : str
        The main directory path containing the dataset.
    annotation : str
        The filename of the COCO annotation JSON file.
    batch_size : int
        Batch size for data loading.
    seed_n : int, optional (default=42)
        Random seed for data splitting.

    Attributes:
    -----------
    main_dir : str
        The main directory path containing the dataset.
    annotation : str
        The filename of the COCO annotation JSON file.
    batch_size : int
        Batch size for data loading.
    seed_n : int
        Random seed for data splitting.
    train_ids : list
        List of image IDs for the training set.
    test_ids : list
        List of image IDs for the validation (or test) set.
    coco_data : dict
        Dictionary containing COCO dataset information.
    """

    def __init__(self, main_dir, annotation, batch_size, seed_n=42):
        super().__init__()
        self.main_dir = main_dir
        self.annotation = annotation
        self.batch_size = batch_size
        self.seed_n = seed_n

    def prepare_data(self):
        pass

    def setup(self, stage=None):
        """
        Sets up the data loader for the specified stage ('fit' or 'test').

        Returns
        ----------
        None
        """
        with open(self.annotation) as json_file:
            self.coco_data = json.load(json_file)
            exclude_video_file_list = [
                "09.08.2023-03-Left",
                "10.08.2023-01-Left",
                "10.08.2023-01-Right",
            ]
            all_ids = []

            for image_info in self.coco_data["images"]:
                image_id = image_info["id"]
                image_file = image_info["file_name"]
                video_file = image_file.split("_")[1]

                if video_file not in exclude_video_file_list:
                    all_ids.append(image_id)

            if stage == "fit" or stage is None:
                train_ids, _ = train_test_split(
                    all_ids,
                    train_size=0.8,
                    shuffle=True,
                    random_state=self.seed_n,
                )
                self.train_ids = train_ids
            if stage == "test" or stage is None:
                _, test_ids = train_test_split(
                    all_ids,
                    test_size=0.2,
                    shuffle=True,
                    random_state=self.seed_n,
                )
                self.test_ids = test_ids

    def train_dataloader(self):
        """
        Returns the data loader for the validation (or test) set.

        Returns
        ----------
        None
        """
        file_paths = [
            next(
                (
                    info
                    for info in self.coco_data["images"]
                    if info["id"] == image_id
                ),
                None,
            )["file_name"]
            for image_id in self.train_ids
        ]
        train_dataset = myFasterRCNNDataset(
            self.main_dir,
            file_paths,
            self.annotation,
            transforms=get_train_transform(),
        )
        return DataLoader(
            train_dataset,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=4,
            collate_fn=collate_fn,
            persistent_workers=True,
        )

    def val_dataloader(self):
        """
        Returns the data loader for the evaluation set.

        Returns
        ----------
        None
        """
        file_paths = [
            next(
                (
                    info
                    for info in self.coco_data["images"]
                    if info["id"] == image_id
                ),
                None,
            )["file_name"]
            for image_id in self.test_ids
        ]
        val_dataset = myFasterRCNNDataset(
            self.main_dir,
            file_paths,
            self.annotation,
            transforms=get_eval_transform(),
        )
        return DataLoader(
            val_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=4,
            collate_fn=collate_fn,
        )
