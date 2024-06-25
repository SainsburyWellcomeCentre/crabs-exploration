import argparse
import ast
import os
import sys
from pathlib import Path

import lightning
import yaml  # type: ignore

from crabs.detection_tracking.datamodules import CrabsDataModule
from crabs.detection_tracking.detection_utils import (
    prep_annotation_files,
    prep_img_directories,
    set_mlflow_run_name,
    setup_mlflow_logger,
    slurm_logs_as_artifacts,
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

    """

    def __init__(self, args: argparse.Namespace) -> None:
        # inputs
        self.args = args
        self.config_file = args.config_file
        self.load_config_yaml()

        # trained model
        self.checkpoint_path = args.checkpoint_path

        # dataset: retrieve from ckpt if possible
        # maybe a different name?
        self.images_dirs = self.get_img_directories_from_ckpt()  # if defined
        # maybe in detection utils? - I think good here cause nothing else uses it for now
        self.annotation_files = (
            self.get_annotation_files_from_ckpt()
        )  # if defined
        self.seed_n = self.get_seed_from_ckpt()

        # Hardware
        self.accelerator = args.accelerator  # --------

        # MLflow
        self.experiment_name = args.experiment_name
        self.mlflow_folder = args.mlflow_folder

        # Debugging
        self.fast_dev_run = args.fast_dev_run
        self.limit_test_batches = args.limit_test_batches

    def load_config_yaml(self):
        with open(self.config_file, "r") as f:
            self.config = yaml.safe_load(f)

    def get_mlflow_client_from_ckpt(self):
        # we assume an mlflow ckpt

        import mlflow

        # roughly assert the format of the path
        assert Path(self.checkpoint_path).parent.stem == "checkpoints"

        # get mlruns path, experiment and run ID associated to this checkpoint
        self.ckpt_mlruns_path = str(Path(self.checkpoint_path).parents[3])
        self.ckpt_experimentID = Path(self.checkpoint_path).parents[2].stem
        self.ckpt_runID = Path(self.checkpoint_path).parents[1].stem

        # create an Mlflow client to interface with mlflow runs
        self.mlrun_client = mlflow.tracking.MlflowClient(
            tracking_uri=self.ckpt_mlruns_path,
        )

        # get params of the run
        run = self.mlrun_client.get_run(self.ckpt_runID)
        params = run.data.params

        return params

    def get_img_directories_from_ckpt(self) -> list[str]:
        # if dataset_dirs is empty: retrieve from ckpt path
        # We assume we always pass a mlflow chckpoint
        # would this work with a remote?
        if not self.args.dataset_dirs:
            # get mlflow client for the ml-runs folder containing the checkpoint
            params = self.get_mlflow_client_from_ckpt()

            # get dataset_dirs used in training job
            train_cli_dataset_dirs = ast.literal_eval(
                params["cli_args/dataset_dirs"]
            )

            # pass that to prep image directories
            images_dirs = prep_img_directories(train_cli_dataset_dirs)

        # if not empty, call the regular one
        else:
            images_dirs = prep_img_directories(self.args.dataset_dirs)

        return images_dirs

    def get_annotation_files_from_ckpt(self) -> list[str]:
        # if no annotation files passed:
        # retrieve from checkpoint
        # pdb.set_trace()
        if not self.args.annotation_files:
            # get mlflow client for the ml-runs folder containing the checkpoint
            params = self.get_mlflow_client_from_ckpt()

            train_cli_dataset_dirs = ast.literal_eval(
                params["cli_args/dataset_dirs"]
            )
            train_cli_annotation_files = ast.literal_eval(
                params["cli_args/annotation_files"]
            )

            annotation_files = prep_annotation_files(
                train_cli_annotation_files, train_cli_dataset_dirs
            )

        else:
            annotation_files = prep_annotation_files(
                self.args.annotation_files, self.args.dataset_dirs
            )
        return annotation_files

    def get_seed_from_ckpt(self) -> int:
        if not self.args.seed_n:
            # get mlflow client for the ml-runs folder containing the checkpoint
            params = self.get_mlflow_client_from_ckpt()

            seed_n = ast.literal_eval(params["cli_args/seed_n"])
        else:
            seed_n = self.args.seed_n
        return seed_n

    def setup_trainer(self):
        """
        Setup trainer object with logging for testing.
        """

        # Assign run name
        self.run_name = set_mlflow_run_name()

        # Setup logger
        mlf_logger = setup_mlflow_logger(
            experiment_name=self.experiment_name,  # "Sep2023_evaluation",
            run_name=self.run_name,
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
        trained_model = FasterRCNN.load_from_checkpoint(self.checkpoint_path)

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
                self.config["score_threshold"],
            )

        # if this is a slurm job: add slurm logs as artifacts
        slurm_job_id = os.environ.get("SLURM_JOB_ID")
        if slurm_job_id:
            slurm_logs_as_artifacts(trainer.logger, slurm_job_id)


def main(args) -> None:
    """
    Main function to orchestrate the testing process.

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
        "--checkpoint_path",
        type=str,
        required=True,  # --------- can we pass experiment and run-id?
        help="Location of trained model",
    )
    parser.add_argument(
        "--config_file",
        type=str,
        default=str(Path(__file__).parent / "config" / "faster_rcnn.yaml"),
        help=(
            "Location of YAML config to control evaluation. "
            "Default: crabs-exploration/crabs/detection_tracking/config/faster_rcnn.yaml"
        ),
    )
    parser.add_argument(
        "--dataset_dirs",
        nargs="+",
        default=[],  # required=True,
        help=(
            "List of dataset directories. If none provided, the ones used for "
            "the ckpt training are used."
        ),
    )
    parser.add_argument(
        "--annotation_files",
        nargs="+",
        default=[],
        help=(
            "List of paths to annotation files. The full path or the filename can be provided. "
            "If only filename is provided, it is assumed to be under dataset/annotations."
            "If none is provided, the annotations from the dataset of the checkpoint are used."
        ),
    )
    parser.add_argument(
        "--seed_n",
        type=int,
        # default=42,
        help=(
            "Seed for dataset splits. If none is provided, the seed from the dataset of "
            "the checkpoint is used."  # No default
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
        "--experiment_name",
        type=str,
        default="Sept2023_evaluation",
        help=(
            "Name of the experiment in MLflow, under which the current run will be logged. "
            "For example, the name of the dataset could be used, to group runs using the same data. "
            "Default: Sept2023_evaluation"
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
            "Debugging option to run training on a fraction of the training set."
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
