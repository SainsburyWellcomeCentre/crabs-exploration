from typing import Optional

import torch
import torchvision
import torchvision.transforms.v2 as transforms
from lightning import LightningDataModule
from torch.utils.data import DataLoader, random_split

from crabs.detection_tracking.datasets import CrabsCocoDetection


class CrabsDataModule(LightningDataModule):
    """A Lightning DataModule class for the crabs data.

    It encapsulate all the steps needed to process the data:
    - computing dataset splits
    - defining the data augmentation transforms
    - defining the dataloaders
    """

    def __init__(
        self,
        list_img_dirs: list[str],
        list_annotation_files: list[str],
        config: dict,
        split_seed: Optional[int] = None,
    ):
        super().__init__()
        self.list_img_dirs = list_img_dirs
        self.list_annotation_files = list_annotation_files
        self.split_seed = split_seed
        self.config = config

    def _get_train_transform(self) -> torchvision.transforms:
        """Define data augmentation transforms for the train set.

        Using transforms.v2, for more details see:
        https://pytorch.org/vision/stable/transforms.html#v1-or-v2-which-one-should-i-use
        https://pytorch.org/vision/main/auto_examples/transforms/plot_transforms_e2e.html#transforms

        """
        jitter = transforms.ColorJitter(
            brightness=self.config["transform_brightness"],
            hue=self.config["transform_hue"],
        )
        gauss = transforms.GaussianBlur(
            kernel_size=self.config["gaussian_blur_params"]["kernel_size"],
            sigma=self.config["gaussian_blur_params"]["sigma"],
        )
        todtype = transforms.ToDtype(torch.float32, scale=True)
        train_transforms = [transforms.ToImage(), jitter, gauss, todtype]
        return transforms.Compose(train_transforms)

    def _get_test_val_transform(self) -> torchvision.transforms:
        """Define data augmentation transforms for the test set.

        Using transforms.v2, for more details see:
        https://pytorch.org/vision/stable/transforms.html#v1-or-v2-which-one-should-i-use
        https://pytorch.org/vision/main/auto_examples/transforms/plot_transforms_e2e.html#transforms

        """
        test_transforms = [
            transforms.ToImage(),
            transforms.ToDtype(torch.float32, scale=True),
        ]
        return transforms.Compose(test_transforms)

    def _collate_fn(self, batch: tuple) -> tuple:
        """Collate function used for dataloaders.

        A custom function is needed for detection
        because the number of bounding boxes varies
        between images of the same batch.
        See https://pytorch.org/vision/main/auto_examples/transforms/plot_transforms_e2e.html#data-loading-and-training-loop

        Parameters
        ----------
        batch : tuple
            a tuple of 2 tuples, the first one holding all images in the batch,
            and the second one holding the corresponding annotations.

        Returns
        -------
        tuple
            a tuple of length = batch size, made up of (image, annotations)
            tuples.
        """
        return tuple(zip(*batch))

    def _compute_splits(
        self,
        transforms_all_splits: torchvision.transforms,
    ) -> tuple[CrabsCocoDetection, CrabsCocoDetection, CrabsCocoDetection]:
        """Compute train/test/validation splits.

        NOTE: The same transforms are passed to all splits.

        The split is reproducible if the same seed is passed.

        The fraction of samples to use in the train set is defined by
        the `train_fraction` in the config file.

        The remaining samples are split between test and validation.
        The fraction of samples to use for validation, over the total
        of samples to use for train+validation is defined by the
        `val_over_test_fraction` parameter in the config file.

        Returns
        -------
        tuple
            A tuple with the train, test and validation datasets
        """

        # Optionally fix the generator for a reproducible split of data
        generator = None
        if self.split_seed:
            generator = torch.Generator().manual_seed(self.split_seed)

        # Create dataset (combining all datasets passed)
        full_dataset = CrabsCocoDetection(
            self.list_img_dirs,
            self.list_annotation_files,
            transforms=transforms_all_splits,
            list_exclude_files=self.config.get(
                "exclude_video_file_list"
            ),  # get value only if key exists
        )

        # Split data into train and test-val sets
        # TODO: split based on video
        train_dataset, test_val_dataset = random_split(
            full_dataset,
            [self.config["train_fraction"], 1 - self.config["train_fraction"]],
            generator=generator,
        )

        # Split test/val sets from the remainder
        test_dataset, val_dataset = random_split(
            test_val_dataset,
            [
                1 - self.config["val_over_test_fraction"],
                self.config["val_over_test_fraction"],
            ],
        )

        return train_dataset, test_dataset, val_dataset

    def prepare_data(self):
        """
        To download data, IO, etc. Useful with shared filesystems,
        only called on 1 GPU/TPU in distributed.
        """
        pass

    def setup(self, stage: str):
        """Setup the data for training, testing and validation.

        Define the transforms for each split of the data and compute them.
        """

        # Assign transforms
        # right now assuming validation and test get the same transforms
        test_and_val_transform = self._get_test_val_transform()
        self.train_transform = self._get_train_transform()
        self.test_transform = test_and_val_transform
        self.val_transform = test_and_val_transform

        # Assign datasets
        self.train_dataset, _, _ = self._compute_splits(self.train_transform)
        _, self.test_dataset, self.val_dataset = self._compute_splits(
            test_and_val_transform
        )

    def train_dataloader(self) -> DataLoader:
        """Define dataloader for the training set"""
        return DataLoader(
            self.train_dataset,
            batch_size=self.config["batch_size_train"],
            shuffle=True,  # a shuffled sampler will be constructed
            num_workers=self.config["num_workers"],
            collate_fn=self._collate_fn,
            persistent_workers=True
            if self.config["num_workers"] > 0
            else False,
            multiprocessing_context="fork"
            if self.config["num_workers"] > 0
            and torch.backends.mps.is_available()
            else None,  # see https://github.com/pytorch/pytorch/issues/87688
        )

    def val_dataloader(self) -> DataLoader:
        """Define dataloader for the validation set"""
        return DataLoader(
            self.val_dataset,
            batch_size=self.config["batch_size_val"],
            shuffle=False,
            num_workers=self.config["num_workers"],
            collate_fn=self._collate_fn,
            persistent_workers=True
            if self.config["num_workers"] > 0
            else False,
        )

    def test_dataloader(self) -> DataLoader:
        """Define dataloader for the test set"""
        return DataLoader(
            self.test_dataset,
            batch_size=self.config["batch_size_test"],
            shuffle=False,
            num_workers=self.config["num_workers"],
            collate_fn=self._collate_fn,
        )
