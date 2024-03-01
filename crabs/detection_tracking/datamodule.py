import json
from typing import Any, List, Optional, Tuple

import lightning as pl
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader
from torchvision.transforms import transforms

from crabs.detection_tracking.custom_faster_rcnn_dataset import (
    CustomFasterRCNNDataset,
)


def collate_fn(batch: List[Any]) -> Optional[Tuple[List[Any], ...]]:
    """
    Collates a batch of input samples into the format expected by PyTorch.

    This function takes a list of samples, filters out any `None` values, and groups
    the remaining samples into a tuple of lists, where each list contains the elements
    from the corresponding position in input batch of the samples.

    Parameters
    ----------
    batch : list
        A list of samples.

    Returns
    -------
    collated : Optional[Tuple[List[Any]]]
        A tuple of lists, where each list contains the elements from the corresponding
        position in the input batch of samples. If the input batch is empty or contains only
        `None` values, the function returns empty tuple.

    Example
    -------
    >>> collate_fn([[1, 2, 3], [4, 5, 6], None, [7, 8, 9]])
    ([1, 4, 7], [2, 5, 8], [3, 6, 9])
    """
    batch = [sample for sample in batch if sample is not None]

    if len(batch) == 0:
        return ()

    return tuple(zip(*batch))


def get_train_transform(config: dict) -> transforms.Compose:
    """
    Get the transform function to apply to an input sample during training.
    This function returns a composed transformation for use with training data.
    The following transforms are applied in sequence:
    1. Apply color jittering with brightness and hue adjustment.
    2. Apply Gaussian blur with specific kernel size and sigma.
        For example kernel size of (5, 9) and sigma (0.1, 5.0).
        This value can be changed in the config file
    3. Convert the image to a PyTorch tensor.

    Parameters
    ----------
    config : dict
        dictionary containing parameters.

    Returns
    -------
    transforms.Compose
        A composed transformation that includes the specified operations.

    Examples
    --------
    >>> train_transform = get_train_transform()
    >>> dataset = MyDataset(transform=train_transform)
    """
    # TODO: testing with different transforms
    custom_transforms = []
    custom_transforms.append(
        transforms.ColorJitter(
            brightness=config["transform_brightness"],
            hue=config["transform_hue"],
        )
    )
    custom_transforms.append(
        transforms.GaussianBlur(
            kernel_size=config["gaussian_blur_params"]["kernel_size"],
            sigma=config["gaussian_blur_params"]["sigma"],
        )
    )
    custom_transforms.append(transforms.ToTensor())

    return transforms.Compose(custom_transforms)


def get_test_transform() -> transforms.Compose:
    """
    Get the transform function to apply to an input sample
    during test (evaluation) and inference.

    Returns
    -------
    transforms.Compose
        A composed transformation that includes the specified operations.
    """
    custom_transforms = []
    custom_transforms.append(transforms.ToTensor())

    return transforms.Compose(custom_transforms)


class CustomDataModule(pl.LightningDataModule):
    """
    Data module for handling data loading and preprocessing for object detection models.

    Parameters:
    -----------
    main_dir : str
        The main directory path containing the dataset.
    annotation : str
        The filename of the COCO annotation JSON file.
    config : dict
        dictionary containing parameters.
    seed_n : int
        Random seed for data splitting.
    """

    def __init__(self, main_dirs, annotations, config, seed_n):
        super().__init__()
        self.main_dirs = main_dirs
        # print(self.main_dirs)
        self.annotations = annotations

        self.config = config
        self.seed_n = seed_n

    def prepare_data(self):
        """
        Function to prepare the data (i.e., data downloading)
        """
        pass

    def setup(self, stage):
        """
        Sets up the data loader for the specified stage
        'fit' for training stage or 'test' for evaluation stage.
        If stage is not specified (i.e., stage is None),
        both blocks of code will execute, which means that the method
        will set up both the training and testing data loaders.

        Returns
        ----------
        None
        """
        self.train_ids = []
        self.test_ids = []
        self.train_file_paths = []
        self.test_file_paths = []

        for main_dir, annotation in zip(self.main_dirs, self.annotations):
            with open(annotation) as json_file:
                coco_data = json.load(json_file)
                exclude_video_file_list = self.config[
                    "exclude_video_file_list"
                ]
                all_ids = []

                for image_info in coco_data["images"]:
                    image_id = image_info["id"]
                    image_file = image_info["file_name"]
                    parts = image_file.split("_")
                    video_file = "_".join(parts[:3])

                    if video_file not in exclude_video_file_list:
                        all_ids.append(image_id)

                if stage == "fit":
                    train_ids, _ = train_test_split(
                        all_ids,
                        train_size=1 - self.config["test_size"],
                        shuffle=True,
                        random_state=self.seed_n,
                    )
                    self.train_ids.extend(train_ids)
                    train_file_paths = [
                        f"{main_dir}/frames/{image_info['file_name']}"
                        for image_info in coco_data["images"]
                        if image_info["id"] in train_ids
                    ]
                    self.train_file_paths.extend(train_file_paths)

                elif stage == "test":
                    _, test_ids = train_test_split(
                        all_ids,
                        test_size=self.config["test_size"],
                        shuffle=True,
                        random_state=self.seed_n,
                    )
                    self.test_ids.extend(test_ids)
                    test_file_paths = [
                        f"{main_dir}/frames/{image_info['file_name']}"
                        for image_info in coco_data["images"]
                        if image_info["id"] in test_ids
                    ]
                    self.test_file_paths.extend(test_file_paths)

    def train_dataloader(self) -> DataLoader:
        """
        Returns the data loader for the training set.

        Returns
        -------
        DataLoader
            DataLoader for the training set.
        """

        train_dataset = CustomFasterRCNNDataset(
            self.train_file_paths,
            self.annotations,  # Assuming all annotations are in the first annotation file
            transforms=get_train_transform(self.config),
        )
        print(len(train_dataset))

        return DataLoader(
            train_dataset,
            batch_size=self.config["batch_size"],
            shuffle=True,
            num_workers=self.config["num_workers"],
            collate_fn=collate_fn,
            persistent_workers=True,
        )

    def test_dataloader(self) -> DataLoader:
        """
        Returns the data loader for the evaluation set.

        Returns
        -------
        DataLoader
            DataLoader for the evaluation set.
        """
        test_dataset = CustomFasterRCNNDataset(
            self.test_file_paths,
            self.annotations,  # Assuming all annotations are in the first annotation file
            transforms=get_test_transform(),
        )
        return DataLoader(
            test_dataset,
            batch_size=self.config["batch_size_test"],
            shuffle=False,
            num_workers=self.config["num_workers"],
            collate_fn=collate_fn,
        )
