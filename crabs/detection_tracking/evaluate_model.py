import argparse
import ast
import logging
import os
import sys
from pathlib import Path

import lightning
import torch
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
        # CLI inputs
        self.args = args

        # trained model
        self.trained_model_path = args.trained_model_path
        self.config_file = args.config_file
        self.get_config_from_ckpt()
        # self.load_config_yaml()  # adds self.config from yaml ----> instead get from ckpt!

        # dataset: retrieve from ckpt if possible
        self.images_dirs = self.get_img_directories_from_ckpt()  # if defined
        self.annotation_files = self.get_annotation_files_from_ckpt()
        self.seed_n = self.get_cli_arg_from_ckpt("seed_n")

        # Hardware
        self.accelerator = args.accelerator

        # MLflow
        self.experiment_name = args.experiment_name
        self.mlflow_folder = args.mlflow_folder

        # Debugging
        self.fast_dev_run = args.fast_dev_run
        self.limit_test_batches = args.limit_test_batches

        logging.info(f"Images directories: {self.images_dirs}")
        logging.info(f"Annotation files: {self.annotation_files}")
        logging.info(f"Seed: {self.seed_n}")

    def get_mlflow_parameters_from_ckpt(self):
        """Get MLflow client from ckpt path and associated hparams"""
        import mlflow

        # roughly assert the format of the path
        assert Path(self.trained_model_path).parent.stem == "checkpoints"

        # get mlruns path, experiment and run ID associated to this checkpoint
        self.ckpt_mlruns_path = str(Path(self.trained_model_path).parents[3])
        self.ckpt_experimentID = Path(self.trained_model_path).parents[2].stem
        self.ckpt_runID = Path(self.trained_model_path).parents[1].stem

        # create an Mlflow client to interface with mlflow runs
        self.mlrun_client = mlflow.tracking.MlflowClient(
            tracking_uri=self.ckpt_mlruns_path,
        )

        # get params of the run
        run = self.mlrun_client.get_run(self.ckpt_runID)
        params = run.data.params

        return params

    def get_config_from_ckpt(self):
        """Get config from checkpoint if not passed as CLI arg"""

        # If passed: used passed config
        if self.config_file:
            with open(self.config_file, "r") as f:
                config_dict = yaml.safe_load(f)

        # If not passed: used config from ckpt
        else:
            params = self.get_mlflow_parameters_from_ckpt()  # string-dict

            # create a 1-level dict
            config_dict = {}
            for p in params:
                if p.startswith("config"):
                    config_dict[p.replace("config/", "")] = ast.literal_eval(
                        params[p]
                    )

            # format as a 2-levels nested dict
            # forward slashes indicate a nested dict
            for key in list(config_dict):  # makes a copy of original keys!
                if "/" in key:
                    key_parts = key.split("/")
                    if key_parts[0] not in config_dict:
                        config_dict[key_parts[0]] = {
                            key_parts[1]: config_dict.pop(key)
                        }  # config_dict[key]}  # initialise
                    else:
                        config_dict[key_parts[0]].update(
                            {key_parts[1]: config_dict.pop(key)}
                        )

            # check there are no more levels
            assert all(["/" not in key for key in config_dict])

        self.config = config_dict

    def get_cli_arg_from_ckpt(self, cli_arg_str):
        """Get CLI argument from checkpoint if not defined"""
        if getattr(self.args, cli_arg_str):
            cli_arg = getattr(self.args, cli_arg_str)
        else:
            params = self.get_mlflow_parameters_from_ckpt()

            cli_arg = ast.literal_eval(params[f"cli_args/{cli_arg_str}"])

        return cli_arg

    def get_img_directories_from_ckpt(self) -> list[str]:
        """Get image directories from checkpoint if not defined."""
        # Get dataset directories from ckpt if not defined
        dataset_dirs = self.get_cli_arg_from_ckpt("dataset_dirs")

        # Extract image directories
        images_dirs = prep_img_directories(dataset_dirs)

        return images_dirs

    def get_annotation_files_from_ckpt(self) -> list[str]:
        """Get annotation files from checkpoint if not defined.

        If annotation_files is not pass as CLI arg to evaluate:
        retrieve annotation_files from ckpt path.
        """

        # Get path to input annotation files from ckpt if not defined
        input_annotation_files = self.get_cli_arg_from_ckpt("annotation_files")

        # Get dataset dirs from ckpt if not defined
        dataset_dirs = self.get_cli_arg_from_ckpt("dataset_dirs")

        # Extract annotation files
        annotation_files = prep_annotation_files(
            input_annotation_files, dataset_dirs
        )
        return annotation_files

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
        "--trained_model_path",
        type=str,
        required=True,  # --------- can we pass experiment and run-id?
        help="Location of trained model (a .ckpt file)",
    )
    parser.add_argument(
        "--config_file",
        type=str,
        default="",
        help=(
            "Location of YAML config to control evaluation. "
            " If None is povided, the config used to train the model is used (recommended)."
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
            "If none are provided (recommended), the annotations from the dataset of the trained model are used."
            "The full path or the filename can be provided. "
            "If only filename is provided, it is assumed to be under dataset/annotations."
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
    torch.set_float32_matmul_precision("medium")

    eval_args = evaluate_parse_args(sys.argv[1:])
    main(eval_args)


if __name__ == "__main__":
    app_wrapper()
