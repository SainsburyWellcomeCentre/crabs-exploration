import argparse
import sys
from pathlib import Path

import lightning
import yaml  # type: ignore
from lightning.pytorch.loggers import MLFlowLogger

from crabs.detection_tracking.datamodules import CrabsDataModule
from crabs.detection_tracking.detection_utils import (
    log_metadata_to_logger,
    prep_annotation_files,
    prep_img_directories,
    set_mlflow_run_name,
    setup_mlflow_logger,
)
from crabs.detection_tracking.models import FasterRCNN
from crabs.detection_tracking.visualization import save_images_with_boxes


class DetectorEvaluation:
    """
    A class for evaluating an object detector using trained model.

    Parameters
    ----------
    args : argparse
        Command-line arguments containing configuration settings.
    config_file : str
        Path to the directory containing configuration file.
    images_dirs : list[str]
        list of paths to the image directories of the datasets.
    annotation_files : list[str]
        list of filenames for the COCO annotations.
    score_threshold : float
        The score threshold for confidence detection.
    ious_threshold : float
        The ious threshold for detection bounding boxes.
    evaluate_dataloader:
        The DataLoader for the test dataset.
    """

    def __init__(
        self,
        args: argparse.Namespace,
    ) -> None:
        self.args = args
        self.config_file = args.config_file
        self.images_dirs = prep_img_directories(args.dataset_dirs)
        self.annotation_files = prep_annotation_files(
            args.annotation_files, args.dataset_dirs
        )
        self.seed_n = args.seed_n
        self.ious_threshold = args.ious_threshold
        self.score_threshold = args.score_threshold
        self.load_config_yaml()

    def load_config_yaml(self):
        with open(self.config_file, "r") as f:
            self.config = yaml.safe_load(f)

    def set_run_name(self):
        self.run_name = set_mlflow_run_name()

    def setup_logger(self) -> MLFlowLogger:
        """
        Setup MLflow logger for testing.

        Includes logging metadata about the job (CLI arguments and SLURM job IDs).
        """
        # Assign run name
        self.set_run_name()

        # Setup logger (no checkpointing)
        mlf_logger = setup_mlflow_logger(
            "evaluation",
            self.run_name,
        )

        # Log metadata to logger: CLI arguments and SLURM (if required)
        mlf_logger = log_metadata_to_logger(mlf_logger, self.args)

        return mlf_logger

    def setup_trainer(self):
        """
        Setup trainer object with logging for testing.
        """

        # Get MLflow logger
        mlf_logger = self.setup_logger()

        # Return trainer linked to logger
        return lightning.Trainer(
            accelerator=self.args.accelerator,
            logger=mlf_logger,
        )

    def evaluate_model(self) -> None:
        """
        Evaluate the trained model on the test dataset.
        """
        # Create datamodule
        data_module = CrabsDataModule(
            self.images_dirs,
            self.annotation_files,
            self.config,
            self.seed_n,
        )

        # Get trained model
        trained_model = FasterRCNN.load_from_checkpoint(
            self.args.checkpoint_path
        )

        # Run testing
        trainer = self.setup_trainer()
        trainer.test(
            trained_model,
            data_module,
        )

        # Save images if required
        if self.args.save_frames:
            save_images_with_boxes(
                data_module.test_dataloader(),
                trained_model,
                self.score_threshold,
            )


def main(args) -> None:
    """
    Main function to orchestrate the testing process using Detector_Test.

    Parameters
    ----------
    args : argparse
        Arguments or configuration settings for testing.

    Returns
    -------
        None
    """
    evaluator = DetectorEvaluation(args)
    evaluator.evaluate_model()


def evaluate_parse_args(args):
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--dataset_dirs",
        nargs="+",
        required=True,
        help="List of dataset directories",
    )
    parser.add_argument(
        "--annotation_files",
        nargs="+",
        default=[],
        help=(
            "List of paths to annotation files. The full path or the filename can be provided. "
            "If only filename is provided, it is assumed to be under dataset/annotations."
        ),
    )
    parser.add_argument(
        "--checkpoint_path",
        type=str,
        required=True,
        help="Location of trained model",
    )
    parser.add_argument(
        "--config_file",
        type=str,
        default=str(Path(__file__).parent / "config" / "faster_rcnn.yaml"),
        help=(
            "Location of YAML config to control training. "
            "Default: crabs-exploration/crabs/detection_tracking/config/faster_rcnn.yaml"
        ),
    )
    parser.add_argument(
        "--accelerator",
        type=str,
        default="gpu",
        help=(
            "Accelerator for Pytorch Lightning. Valid inputs are: cpu, gpu, tpu, ipu, auto, mps. Default: gpu."
            "See https://lightning.ai/docs/pytorch/stable/common/trainer.html#accelerator "
            "and https://lightning.ai/docs/pytorch/stable/accelerators/mps_basic.html#run-on-apple-silicon-gpus"
        ),
    )
    parser.add_argument(
        "--score_threshold",
        type=float,
        default=0.1,
        help="Threshold for confidence score. Default: 0.1",
    )
    parser.add_argument(
        "--ious_threshold",
        type=float,
        default=0.1,
        help="Threshold for IOU. Default: 0.1",
    )
    parser.add_argument(
        "--seed_n",
        type=int,
        default=42,
        help="Seed for dataset splits. Default: 42",
    )
    parser.add_argument(
        "--save_frames",
        action="store_true",
        help=("Save predicted frames with bounding boxes."),
    )
    return parser.parse_args(args)


def app_wrapper():
    eval_args = evaluate_parse_args(sys.argv[1:])
    main(eval_args)


if __name__ == "__main__":
    app_wrapper()
