import lightning as L
import torch
from torch.utils.data import DataLoader, random_split
from torchvision.transforms import transforms

# from torchvision.transforms import v2
from crabs.detection_tracking.datasets import CrabsCocoDetection


class CrabsDataModule(L.LightningDataModule):
    def __init__(
        self, imgs_path: str, annotations_path, config, split_seed=None
    ):
        super().__init__()
        self.imgs_path = imgs_path
        self.annotations_path = annotations_path
        self.config = config
        self.split_seed = split_seed

    def _get_train_transform(self):
        # see https://pytorch.org/vision/stable/transforms.html#v1-or-v2-which-one-should-i-use
        # https://pytorch.org/vision/main/auto_examples/transforms/plot_transforms_e2e.html#transforms
        train_transforms = []
        train_transforms.append(
            transforms.ColorJitter(
                brightness=self.config["transform_brightness"],
                hue=self.config["transform_hue"],
            )
        )
        train_transforms.append(
            transforms.GaussianBlur(
                kernel_size=self.config["gaussian_blur_params"]["kernel_size"],
                sigma=self.config["gaussian_blur_params"]["sigma"],
            )
        )
        train_transforms.append(
            transforms.GaussianBlur(
                kernel_size=self.config["gaussian_blur_params"]["kernel_size"],
                sigma=self.config["gaussian_blur_params"]["sigma"],
            )
        )
        train_transforms.append(transforms.ToTensor())  # ToImage()?
        return train_transforms

    def _get_test_val_transform(self):
        test_transforms = []
        test_transforms.append(transforms.ToTensor())
        return test_transforms

    # def _collate_fn(batch):
    #     # https://pytorch.org/vision/main/auto_examples/transforms/plot_transforms_e2e.html#data-loading-and-training-loop
    #     # We need a custom fn because the number of bounding boxes varies between images of the same batch
    #     # why would a sample be None?
    #     batch = [sample for sample in batch if sample is not None]

    #     if len(batch) == 0:
    #         return tuple()

    #     return tuple(zip(*batch))

    def _compute_splits(self):
        # Compute train/test/val splits
        # - define split
        # - make shuffles if required? -- via seed? log in mlflow?
        # - exclude relevant files

        # Optionally fix the generator for reproducible results
        generator = None
        if self.split_seed:
            generator = torch.Generator().manual_seed(self.split_seed)

        # Instantiate dataset
        # exclude files here? or when creating the dataset?
        full_dataset = CrabsCocoDetection(
            self.imgs_path,
            self.annotations_path,
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
            [0.5, 0.5],  # can I pass zero here?
        )

        return train_dataset, test_dataset, val_dataset

    def prepare_data(self):
        # download, IO, etc. Useful with shared filesystems
        # only called on 1 GPU/TPU in distributed
        pass

    def setup(self, stage: str):
        """Define transforms and assign train/val datasets for use in dataloaders.

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
        # split & transforms
        # make assignments here (val/train/test split)
        # called on every process in DDP (?)

        # Assign transforms
        self.train_transform = self._get_train_transform()
        self.test_transform = self._get_test_val_transform()
        self.val_transform = self._get_test_val_transform()

        # Assign datasets for dataloader depending on stage
        # -- omit predict for now
        train_dataset, test_dataset, val_dataset = self._compute_splits()
        if stage == "fit":
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
            collate_fn=lambda batch: tuple(
                zip(*batch)
            ),  # self._collate_fn,  # --- why do we need it? Yes
            persistent_workers=True,
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
            collate_fn=lambda batch: tuple(zip(*batch)),
        )

    def test_dataloader(self):
        return DataLoader(
            self.test_dataset,
            batch_size=self.config["batch_size_test"],
            shuffle=False,
            num_workers=self.config["num_workers"],
            collate_fn=lambda batch: tuple(zip(*batch)),
        )
