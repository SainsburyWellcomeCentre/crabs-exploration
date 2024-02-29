import argparse

import lightning as pl
import torch
import yaml  # type: ignore

from crabs.detection_tracking.datamodule import CustomDataModule
from crabs.detection_tracking.detection_utils import save_model
from crabs.detection_tracking.models import FasterRCNN


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
        self.accelerator = args.accelerator
        self.seed_n = args.seed_n
        self.annotation = f"{self.main_dir}/annotations/{self.annotation_file}"
        self.load_config_yaml()

    def load_config_yaml(self):
        with open(self.config_file, "r") as f:
            self.config = yaml.safe_load(f)

    def train_model(self):
        data_module = CustomDataModule(
            self.main_dir, self.annotation, self.config, self.seed_n
        )

        lightning_model = FasterRCNN(self.config)

        mlf_logger = pl.pytorch.loggers.MLFlowLogger(
            experiment_name="lightning_logs", tracking_uri="file:./ml-runs"
        )
        trainer = pl.Trainer(
            max_epochs=self.config["num_epochs"],
            accelerator=self.accelerator,
            logger=mlf_logger,
        )

        trainer.fit(lightning_model, data_module)
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
        "--annotation_file",
        type=str,
        required=True,
        help="filename for coco annotation",
    )
    parser.add_argument(
        "--accelerator",
        type=str,
        default="gpu",
        help="accelerator for pytorch lightning",
    )
    parser.add_argument(
        "--seed_n",
        type=int,
        default=42,
        help="seed for random state",
    )
    args = parser.parse_args()
    torch.set_float32_matmul_precision("medium")
    main(args)
