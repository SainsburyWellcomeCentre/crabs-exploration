# https://pytorch.org/tutorials/beginner/basics/data_tutorial.html#creating-a-custom-dataset-for-your-files

from pathlib import Path

import torch
from PIL import Image
from pycocotools.coco import COCO


class CustomFasterRCNNDataset(torch.utils.data.Dataset):
    def __init__(self, data_dir, annotations, transform=None):
        self.data_dir = data_dir  # parent data to frames and annotations
        self.annotations = [
            COCO(annotation) for annotation in annotations
        ]  # annotations ---can I derive this from the assumed structure of data dir?
        self.transform = transform
        # self.target_transform = target_transform

        self.img_dir = Path(self.data_dir) / "frames"
        self.img_paths = [
            f
            for f in Path(self.img_dir).iterdir()
            if f.is_file()  # and image?
        ]  # is this the full path

        self.img_labels = []

    def __len__(self):
        return len(self.img_labels)

    def __getitem__(self, idx):
        img = Image.open(self.img_paths[idx]).convert("RGB")

        # can I use this?
        # https://pytorch.org/vision/stable/auto_examples/transforms/plot_transforms_e2e.html#dataset-preparation

        return img  # , annotation
