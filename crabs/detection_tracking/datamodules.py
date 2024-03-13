import lightning as L
import torch
import torchvision.transforms.v2 as transforms
from torch.utils.data import DataLoader, random_split

# from torchvision.transforms import v2
from crabs.detection_tracking.datasets import CrabsCocoDetection


class CrabsDataModule(L.LightningDataModule):
    def __init__(
        self,
        list_img_dirs: list[str],
        list_annotation_files: list[str],
        config: dict,
        split_seed=None,
    ):
        super().__init__()
        self.list_img_dirs = list_img_dirs
        self.list_annotation_files = list_annotation_files
        self.split_seed = split_seed
        self.config = config

    def _get_train_transform(self):
        train_transforms = transforms.Compose(
            [
                transforms.ToImage(),
                transforms.ColorJitter(
                    brightness=self.config["transform_brightness"],
                    hue=self.config["transform_hue"],
                ),
                transforms.GaussianBlur(
                    kernel_size=self.config["gaussian_blur_params"][
                        "kernel_size"
                    ],
                    sigma=self.config["gaussian_blur_params"]["sigma"],
                ),
                transforms.ToDtype(torch.float32, scale=True),
            ]
        )

        # train_transforms.append(transforms.ToTensor())  # ToImage()?
        return train_transforms

    def _get_test_val_transform(self):
        # see https://pytorch.org/vision/stable/transforms.html#v1-or-v2-which-one-should-i-use
        # https://pytorch.org/vision/main/auto_examples/transforms/plot_transforms_e2e.html#transforms
        test_transforms = []
        test_transforms.append(transforms.ToTensor())
        return test_transforms

    def _collate_fn(self, batch):
        # https://pytorch.org/vision/main/auto_examples/transforms/plot_transforms_e2e.html#data-loading-and-training-loop
        # We need a custom fn because the number of bounding boxes varies between images of the same batch
        return tuple(zip(*batch))

    def _compute_splits(self):
        # Compute train/test/val splits
        # - define split
        # - make shuffles if required? -- via seed? log in mlflow?
        # - exclude relevant files

        # Optionally fix the generator for a reproducible split of data
        generator = None
        if self.split_seed:
            generator = torch.Generator().manual_seed(self.split_seed)

        # Create dataset (combining all datasets passed)
        full_dataset = CrabsCocoDetection(
            self.list_img_dirs,
            self.list_annotation_files,
            transforms=self.train_transform,
            # exclude_files_w_regex------------------
        )

        # Split data into train/test-val
        # can we specify what to have in train/test?
        train_dataset, test_val_dataset = random_split(
            full_dataset,
            [self.config["train_fraction"], 1 - self.config["train_fraction"]],
            generator=generator,
        )

        # Split test/val in equal parts?
        # define val but not use for now?
        test_dataset, val_dataset = random_split(
            test_val_dataset,
            [
                1 - self.config["val_over_test_fraction"],
                self.config["val_over_test_fraction"],
            ],  # [0.5, 0.5],  # can I pass zero here?
        )

        return train_dataset, test_dataset, val_dataset

    def prepare_data(self):
        """
        To download data, IO, etc. Useful with shared filesystems,
        only called on 1 GPU/TPU in distributed.
        """
        pass

    def setup(self, stage: str):
        """Define transforms for data augmentation and
        assign train/val datasets for use in dataloaders.

        Sets up the data loader for the specified stage
        'fit' for training stage or 'test' for evaluation stage.
        If stage is not specified (i.e., stage is None),
        both blocks of code will execute, which means that the method
        will set up both the training and testing data loaders.

        Parameters
        ----------
        stage : str
            _description_
        config : dict
            _description_
        """

        # Assign transforms
        self.train_transform = self._get_train_transform()
        self.test_transform = self._get_test_val_transform()
        self.val_transform = self._get_test_val_transform()

        # Assign datasets for dataloader depending on stage
        # omitting "predict" stage for now
        train_dataset, test_dataset, val_dataset = self._compute_splits()
        if stage == "fit":  # or stage=='train':
            self.train_dataset = train_dataset
            self.val_dataset = val_dataset

        if stage == "test":
            self.test_dataset = test_dataset

    def train_dataloader(self):
        # https://github.com/pytorch/vision/blob/423a1b0ebdea077cc69478812890845741048d2e/references/detection/train.py#L209
        return DataLoader(
            self.train_dataset,
            batch_size=self.config["batch_size_train"],
            shuffle=True,  # A sequential or shuffled sampler will be automatically constructed based on the shuffle argument
            num_workers=self.config["num_workers"],  # set to auto?
            collate_fn=self._collate_fn,
            persistent_workers=True,
            multiprocessing_context="fork"
            if torch.backends.mps.is_available()
            else None,
            # see https://github.com/pytorch/pytorch/issues/87688
            # --- why do we need it? to use the same workers across epochs (after loader is exhausted)
            # interesting if it takes a lot of time to spawn workers at the start of the epoch
            # if they persist, workers stay with their state
        )

    def val_dataloader(self):
        return DataLoader(
            self.val_dataset,
            batch_size=self.config["batch_size_val"],
            shuffle=False,
            num_workers=self.config["num_workers"],
            collate_fn=self._collate_fn,
        )

    def test_dataloader(self):
        return DataLoader(
            self.test_dataset,
            batch_size=self.config["batch_size_test"],
            shuffle=False,
            num_workers=self.config["num_workers"],
            collate_fn=self._collate_fn,
        )
