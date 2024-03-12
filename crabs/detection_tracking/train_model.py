import argparse
import datetime

import lightning as pl
import torch
import yaml  # type: ignore

from crabs.detection_tracking.datamodules import CrabsDataModule
from crabs.detection_tracking.detection_utils import save_model
from crabs.detection_tracking.models import FasterRCNN


class DectectorTrain:
    """Training class for detector algorithm

    Parameters
    ----------
    args: argparse.Namespace
        An object containing the parsed command-line arguments.

    Attributes
    ----------
    config_file : str
        Path to the directory containing configuration file.
    main_dirs : List[str]
        List of paths to the main directories of the datasets.
    annotation_files : List[str]
        List of filenames for the COCO annotations.
    model_name : str
        The model use to train the detector.
    """

    def __init__(self, args):
        self.config_file = args.config_file
        self.main_dirs = args.main_dir
        self.annotation_files = args.annotation_file
        self.accelerator = args.accelerator
        self.seed_n = args.seed_n
        self.load_config_yaml()

    def load_config_yaml(self):
        with open(self.config_file, "r") as f:
            self.config = yaml.safe_load(f)

    def train_model(self):
        annotations = []
        for main_dir, annotation_file in zip(
            self.main_dirs, self.annotation_files
        ):
            annotations.append(f"{main_dir}/annotations/{annotation_file}")

        data_module = CrabsDataModule(
            self.main_dirs, annotations, self.config, self.seed_n
        )

        timestamp = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        run_name = f"run_{timestamp}"

        mlf_logger = pl.pytorch.loggers.MLFlowLogger(
            run_name=run_name,
            experiment_name=args.experiment_name,
            tracking_uri="file:./ml-runs",
        )

        mlf_logger.log_hyperparams(self.config)

        lightning_model = FasterRCNN(self.config)

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
    trainer = DectectorTrain(args)
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
        nargs="+",
        required=True,
        help="list of locations of images and coco annotations",
    )
    parser.add_argument(
        "--annotation_file",
        nargs="+",
        required=True,
        help="list of filenames for coco annotations",
    )
    parser.add_argument(
        "--accelerator",
        type=str,
        default="gpu",
        help="accelerator for pytorch lightning",
    )
    parser.add_argument(
        "--experiment_name",
        type=str,
        default="Sept2023",
        help="the name for the experiment in MLflow, under which the current run will be logged. For example, the name of the dataset could be used, to group runs using the same data.",
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
