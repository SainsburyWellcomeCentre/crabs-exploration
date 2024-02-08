import argparse
import json

import torch
import pytorch_lightning as pl
import yaml  # type: ignore
from detection_utils import (
    load_dataset,
    save_model,
)
from models import FasterRCNN


class Dectector_Train:
    """Training class for detector algorithm

    Parameters
    ----------
    args: argparse.Namespace
        An object containing the parsed command-line arguments.

    Attributes
    ----------
    config_file : str
        Path to the directory containing configuration file.
    main_dir : str
        Path to the main directory of the dataset.
    model_name : str
        The model use to train the detector.
    """

    def __init__(self, args):
        self.config_file = args.config_file
        self.main_dir = args.main_dir
        self.annotation_file = args.annotation_file
        self.model_name = args.model_name
        self.annotation = f"{self.main_dir}/annotations/{self.annotation_file}"
        self.load_config_yaml()

    def load_config_yaml(self):
        with open(self.config_file, "r") as f:
            self.config = yaml.safe_load(f)

    def train_model(self):
        train_dataloader = load_dataset(
            self.main_dir,
            self.annotation,
            self.config["batch_size"],
            training=True,
        )

        lightning_model = FasterRCNN(self.config)

        trainer = pl.Trainer(max_epochs=self.config["num_epochs"])

        trainer.fit(lightning_model, train_dataloader)
        if self.config["save"]:
            save_model(lightning_model)


def main(args) -> None:
    """
    Main function to orchestrate the training process.

    Parameters
    ----------
    args: argparse.Namespace
        An object containing the parsed command-line arguments.

    Returns
    ----------
    None
    """
    trainer = Dectector_Train(args)
    trainer.train_model()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config_file",
        type=str,
        default="crabs/detection_tracking/config/faster_rcnn.yaml",
        help="location of YAML config to control training",
    )
    parser.add_argument(
        "--main_dir",
        type=str,
        required=True,
        help="location of images and coco annotation",
    )
    parser.add_argument(
        "--model_name",
        type=str,
        default="faster_rcnn",
        help="the model to use to train the object detection.",
    )
    parser.add_argument(
        "--annotation_file",
        type=str,
        required=True,
        help="filename for coco annotation",
    )
    args = parser.parse_args()
    main(args)
