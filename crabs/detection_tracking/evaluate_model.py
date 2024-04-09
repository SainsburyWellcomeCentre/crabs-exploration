import argparse
import datetime
import sys

import lightning
import torch
import yaml  # type: ignore
from lightning.pytorch.loggers import MLFlowLogger

from crabs.detection_tracking.datamodules import CrabsDataModule
from crabs.detection_tracking.detection_utils import (
    prep_annotation_files,
    prep_img_directories,
)
from crabs.detection_tracking.evaluate import (
    save_images_with_boxes,
)
from crabs.detection_tracking.models import FasterRCNN


class DetectorEvaluation:
    """
    A class for evaluating an object detector using trained model.

    Parameters
    ----------
    args : argparse
        Command-line arguments containing configuration settings.
    config_file : str
        Path to the directory containing configuration file.
    images_dirs : List[str]
        List of paths to the main directories of the datasets.
    annotation_files : List[str]
        List of filenames for the COCO annotations.
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
        self.images_dirs = prep_img_directories(args.images_dirs)
        self.annotation_files = prep_annotation_files(
            args.annotation_files, args.images_dirs
        )
        self.ious_threshold = args.ious_threshold
        self.score_threshold = args.score_threshold

    def load_trained_model(self) -> None:
        """
        Load the trained model.

        Returns
        -------
        None
        """
        self.trained_model = torch.load(
            self.args.model_dir,
            map_location=torch.device(self.args.accelerator),
        )

    def evaluate_model(self) -> None:
        """
        Evaluate the trained model on the test dataset.

        Returns
        -------
        None
        """
        with open(self.config_file, "r") as f:
            config = yaml.safe_load(f)

        self.load_trained_model()

        data_module = CrabsDataModule(
            self.images_dirs,
            self.annotation_files,
            config,
            self.args.seed_n,
        )
        data_module.setup("test")

        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        print(timestamp)
        run_name = f"run_{timestamp}"

        mlf_logger = MLFlowLogger(
            run_name=run_name,
            experiment_name="evaluation",
            tracking_uri="file:./ml-runs",
        )

        mlf_logger.log_hyperparams(config)
        mlf_logger.log_hyperparams({"split_seed": self.args.seed_n})
        mlf_logger.log_hyperparams({"cli_args": self.args})

        faster_rcnn_model = FasterRCNN(config)
        faster_rcnn_model.load_state_dict(self.trained_model.state_dict())

        trainer = lightning.Trainer(
            accelerator=self.args.accelerator,
            logger=mlf_logger,
        )
        trainer.test(
            faster_rcnn_model, dataloaders=data_module.test_dataloader()
        )

        if self.args.save_frames:
            save_images_with_boxes(
                data_module.test_dataloader(),
                self.trained_model,
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
        "--config_file",
        type=str,
        default="crabs/detection_tracking/config/faster_rcnn.yaml",
        help="location of YAML config to control training",
    )
    parser.add_argument(
        "--model_dir",
        type=str,
        required=True,
        help="location of trained model",
    )
    parser.add_argument(
        "--images_dirs",
        type=str,
        nargs="+",
        required=True,
        help="list of paths to images directories",
    )
    parser.add_argument(
        "--annotation_files",
        type=str,
        nargs="+",
        required=True,
        help="list of paths to annotation files",
    )
    parser.add_argument(
        "--score_threshold",
        type=float,
        default=0.1,
        help="threshold for confidence score",
    )
    parser.add_argument(
        "--ious_threshold",
        type=float,
        default=0.1,
        help="threshold for IOU",
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
    parser.add_argument(
        "--save_frames",
        action="store_true",
        help=("Save predicted frames with bboxes."),
    )
    return parser.parse_args(args)


if __name__ == "__main__":
    args = evaluate_parse_args(sys.argv[1:])
    main(args)
