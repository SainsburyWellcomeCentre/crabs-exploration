"""Script to evaluate a trained object detector."""

import argparse
import logging
import os
import sys

import lightning
import torch

from crabs.detector.datamodules import CrabsDataModule
from crabs.detector.models import FasterRCNN
from crabs.detector.utils.detection import (
    setup_mlflow_logger,
    slurm_logs_as_artifacts,
)
from crabs.detector.utils.evaluate import (
    get_annotation_files_from_ckpt,
    get_cli_arg_from_ckpt,
    get_config_from_ckpt,
    get_img_directories_from_ckpt,
    get_mlflow_experiment_name_from_ckpt,
    get_mlflow_parameters_from_ckpt,
    get_mlflow_run_name_from_ckpt,
)
from crabs.detector.utils.visualization import save_images_with_boxes


class DetectorEvaluate:
    """Interface for evaluating an object detector.

    Parameters
    ----------
    args : argparse
        Command-line arguments containing configuration settings.

    """

    def __init__(self, args: argparse.Namespace) -> None:
        """Initialise the evaluation interface with the given arguments."""
        # CLI inputs
        self.args = args

        # trained model
        self.trained_model_path = args.trained_model_path
        self.trained_model_run_name = get_mlflow_parameters_from_ckpt(
            self.trained_model_path
        )["run_name"]

        # config: retreieve from ckpt if not passed as CLI argument
        self.config_file = args.config_file
        self.config = get_config_from_ckpt(
            config_file=self.config_file,
            trained_model_path=self.trained_model_path,
        )

        # dataset: retrieve from ckpt if no CLI arguments are passed
        self.images_dirs = get_img_directories_from_ckpt(
            args=self.args, trained_model_path=self.trained_model_path
        )
        self.annotation_files = get_annotation_files_from_ckpt(
            args=self.args, trained_model_path=self.trained_model_path
        )
        self.seed_n = get_cli_arg_from_ckpt(
            args=self.args,
            cli_arg_str="seed_n",
            trained_model_path=self.trained_model_path,
        )

        # Hardware
        self.accelerator = args.accelerator

        # MLflow
        self.experiment_name = get_mlflow_experiment_name_from_ckpt(
            args=self.args, trained_model_path=self.trained_model_path
        )
        self.mlflow_folder = args.mlflow_folder

        # Debugging
        self.fast_dev_run = args.fast_dev_run
        self.limit_test_batches = args.limit_test_batches

        logging.info("Dataset")
        logging.info(f"Images directories: {self.images_dirs}")
        logging.info(f"Annotation files: {self.annotation_files}")
        logging.info(f"Seed: {self.seed_n}")

    def setup_trainer(self):
        """Set up trainer object with logging for testing."""
        # Assign run name
        self.run_name = get_mlflow_run_name_from_ckpt(
            self.args.mlflow_run_name_auto, self.trained_model_run_name
        )

        # Setup logger
        mlf_logger = setup_mlflow_logger(
            experiment_name=self.experiment_name,  # get from ckpt?
            run_name=self.run_name,  # -------------->get from ckpt?
            mlflow_folder=self.mlflow_folder,
            cli_args=self.args,
        )

        # Return trainer linked to logger
        return lightning.Trainer(
            accelerator=self.accelerator,
            logger=mlf_logger,
            fast_dev_run=self.fast_dev_run,
            limit_test_batches=self.limit_test_batches,
        )

    def evaluate_model(self) -> None:
        """Evaluate the trained model on the test dataset."""
        # Create datamodule
        data_module = CrabsDataModule(
            list_img_dirs=self.images_dirs,
            list_annotation_files=self.annotation_files,
            split_seed=self.seed_n,
            config=self.config,
            no_data_augmentation=True,
        )
        # breakpoint()

        # Get trained model
        trained_model = FasterRCNN.load_from_checkpoint(
            self.trained_model_path, config=self.config
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
                test_dataloader=data_module.test_dataloader(),
                trained_model=trained_model,
                output_dir=self.args.frames_output_dir,
                score_threshold=self.args.frames_score_threshold,
            )

        # if this is a slurm job: add slurm logs as artifacts
        slurm_job_id = os.environ.get("SLURM_JOB_ID")
        slurm_job_name = os.environ.get("SLURM_JOB_NAME")
        if slurm_job_id and (slurm_job_name != "bash"):
            slurm_logs_as_artifacts(trainer.logger, slurm_job_id)


def main(args) -> None:
    """Run detector testing.

    Parameters
    ----------
    args : argparse
        Arguments or configuration settings for testing.

    Returns
    -------
        None

    """
    evaluator = DetectorEvaluate(args)
    evaluator.evaluate_model()


def evaluate_parse_args(args):
    """Parse command-line arguments for evaluation."""
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--trained_model_path",
        type=str,
        required=True,
        help="Location of trained model (a .ckpt file)",
    )
    parser.add_argument(
        "--config_file",
        type=str,
        default="",
        help=(
            "Location of YAML config to control evaluation. "
            "If none is povided, the config used to train "
            "the model is used (recommended)."
        ),
    )
    parser.add_argument(
        "--dataset_dirs",
        nargs="+",
        default=[],
        help=(
            "List of dataset directories. "
            "If none is provided (recommended), the datasets used for "
            "the trained model are used."
        ),
    )
    parser.add_argument(
        "--annotation_files",
        nargs="+",
        default=[],
        help=(
            "List of paths to annotation files. "
            "If none are provided (recommended), the annotations "
            "from the dataset of the trained model are used."
            "The full path or the filename can be provided. "
            "If only filename is provided, it is assumed to be "
            "under dataset/annotations."
        ),
    )
    parser.add_argument(
        "--seed_n",
        type=int,
        help=(
            "Seed for dataset splits. "
            "If none is provided (recommended), the seed from the dataset of "
            "the trained model is used."
        ),
    )

    parser.add_argument(
        "--accelerator",
        type=str,
        default="gpu",
        help=(
            "Accelerator for Pytorch Lightning. "
            "Valid inputs are: cpu, gpu, tpu, ipu, auto, mps. Default: gpu."
            "See https://lightning.ai/docs/pytorch/stable/common/trainer.html#accelerator "  # noqa: E501
            "and https://lightning.ai/docs/pytorch/stable/accelerators/mps_basic.html#run-on-apple-silicon-gpus"  # noqa: E501
        ),
    )
    parser.add_argument(
        "--experiment_name",
        type=str,
        help=(
            "Name of the experiment in MLflow, under which the current run "
            "will be logged. "
            "For example, the name of the dataset could be used, to group "
            "runs using the same data. "
            "By default: <experiment_training_job>_evaluation."
        ),
    )
    parser.add_argument(
        "--fast_dev_run",
        action="store_true",
        help="Debugging option to run training for one batch and one epoch",
    )
    parser.add_argument(
        "--limit_test_batches",
        type=float,
        default=1.0,
        help=(
            "Debugging option to run training on a fraction of "
            "the training set."
            "Default: 1.0 (all the training set)"
        ),
    )
    parser.add_argument(
        "--mlflow_folder",
        type=str,
        default="./ml-runs",
        help=("Path to MLflow directory. Default: ./ml-runs"),
    )
    parser.add_argument(
        "--mlflow_run_name_auto",
        action="store_true",
        help=(
            "Set the evaluation run name automatically from MLflow, ignoring the training job run name."
        ),
    )
    parser.add_argument(
        "--save_frames",
        action="store_true",
        help=("Save predicted frames with bounding boxes."),
    )
    parser.add_argument(
        "--frames_score_threshold",
        type=float,
        default=0.5,
        help=(
            "Score threshold for visualising detections on output frames. "
            "Default: 0.5"
        ),
    )
    parser.add_argument(
        "--frames_output_dir",
        type=str,
        default="",
        help=(
            "Output directory for the exported frames. "
            "By default, the frames are saved in a "
            "`results_<timestamp> folder "
            "under the current working directory."
        ),
    )
    return parser.parse_args(args)


def app_wrapper():
    """Wrap function to run the evaluation."""
    torch.set_float32_matmul_precision("medium")

    eval_args = evaluate_parse_args(sys.argv[1:])
    main(eval_args)


if __name__ == "__main__":
    app_wrapper()
