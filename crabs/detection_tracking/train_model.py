import argparse
import datetime
import sys
from pathlib import Path

import lightning as pl
import torch
import yaml  # type: ignore

from crabs.detection_tracking.datamodules import CrabsDataModule
from crabs.detection_tracking.detection_utils import save_model
from crabs.detection_tracking.models import FasterRCNN

DEFAULT_ANNOTATIONS_FILENAME = "VIA_JSON_combined_coco_gen.json"


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
        self.images_dirs = self.prep_img_directories(
            args.dataset_dirs  # args only?
        )  # list of paths
        self.annotation_files = self.prep_annotation_files(
            args.annotation_files, args.dataset_dirs
        )  # list of paths
        self.accelerator = args.accelerator
        self.seed_n = args.seed_n
        self.experiment_name = args.experiment_name
        self.load_config_yaml()

    def load_config_yaml(self):
        with open(self.config_file, "r") as f:
            self.config = yaml.safe_load(f)

    def prep_img_directories(self, dataset_dirs: list[str]):
        images_dirs = []
        for dataset in dataset_dirs:
            images_dirs.append(str(Path(dataset) / "frames"))
        return images_dirs

    def prep_annotation_files(
        self, input_annotation_files: list[str], dataset_dirs: list[str]
    ):
        # prepare list of annotation files
        annotation_files = []

        # if none are passed: assume default filename for annotations,
        # under default location
        if not input_annotation_files:
            for dataset in dataset_dirs:
                annotation_files.append(
                    str(
                        Path(dataset)
                        / "annotations"
                        / DEFAULT_ANNOTATIONS_FILENAME
                    )
                )

        # if a list of annotation files/filepaths is passed
        else:
            for annot, dataset in zip(input_annotation_files, dataset_dirs):
                # if the annotation is only filename:
                # assume file is under 'annotation' directory
                if Path(annot).name == annot:
                    annotation_files.append(
                        str(Path(dataset) / "annotations" / annot)
                    )
                # otherwise assume the full path to the annotations file is passed
                else:
                    annotation_files.append(annot)

        return annotation_files

    def train_model(self):
        # Create data module
        data_module = CrabsDataModule(
            self.images_dirs,  # list of paths
            self.annotation_files,  # list of paths
            self.config,
            self.seed_n,
        )

        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        run_name = f"run_{timestamp}"

        # Initialise MLflow logger
        mlf_logger = pl.pytorch.loggers.MLFlowLogger(
            run_name=run_name,
            experiment_name=self.experiment_name,
            tracking_uri="file:./ml-runs",
        )

        mlf_logger.log_hyperparams(self.config)
        mlf_logger.log_hyperparams({"split_seed": self.seed_n})

        lightning_model = FasterRCNN(self.config)

        trainer = pl.Trainer(
            max_epochs=self.config["num_epochs"],
            accelerator=self.accelerator,
            logger=mlf_logger,
        )

        # Run training
        trainer.fit(lightning_model, data_module)

        # Save model if required
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


def train_parse_args(args):
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config_file",
        type=str,
        default=str(Path(__file__).parent / "config" / "faster_rcnn.yaml"),
        help="location of YAML config to control training",
    )
    parser.add_argument(
        "--dataset_dirs",
        nargs="+",
        required=True,
        help="list of dataset directories",
    )
    parser.add_argument(
        "--annotation_files",
        nargs="+",
        default=[],
        help="list of paths to annotation files. The full path or the filename can be provided. If only filename is provided, it is assumed to be under dataset/annotations.",
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
        help="seed for dataset splits",
    )
    return parser.parse_args(args)


if __name__ == "__main__":
    torch.set_float32_matmul_precision("medium")

    train_args = train_parse_args(sys.argv[1:])
    main(train_args)
