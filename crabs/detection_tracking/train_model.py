import argparse
import os
import sys
from pathlib import Path

import lightning
import torch
import yaml  # type: ignore
from lightning.pytorch.callbacks import ModelCheckpoint

from crabs.detection_tracking.datamodules import CrabsDataModule
from crabs.detection_tracking.detection_utils import (
    prep_annotation_files,
    prep_img_directories,
    set_mlflow_run_name,
    setup_mlflow_logger,
    slurm_logs_as_artifacts,
)
from crabs.detection_tracking.models import FasterRCNN
from crabs.detection_tracking.optuna_fn import optimize_hyperparameters


class DectectorTrain:
    """Training class for detector algorithm

    Parameters
    ----------
    args: argparse.Namespace
        An object containing the parsed command-line arguments.
    """

    def __init__(self, args):
        # inputs
        self.args = args
        self.config_file = args.config_file
        self.load_config_yaml()

        # dataset
        self.images_dirs = prep_img_directories(args.dataset_dirs)
        self.annotation_files = prep_annotation_files(
            args.annotation_files, args.dataset_dirs
        )
        self.seed_n = args.seed_n

        # Hardware
        self.accelerator = args.accelerator

        # MLflow
        self.experiment_name = args.experiment_name
        self.mlflow_folder = args.mlflow_folder

        # Debugging
        self.fast_dev_run = args.fast_dev_run
        self.limit_train_batches = args.limit_train_batches

    def load_config_yaml(self):
        with open(self.config_file, "r") as f:
            self.config = yaml.safe_load(f)

    def setup_trainer(self):
        """
        Setup trainer with logging and checkpointing.
        """
        self.run_name = set_mlflow_run_name()

        # Setup logger with checkpointing
        mlf_logger = setup_mlflow_logger(
            experiment_name=self.experiment_name,
            run_name=self.run_name,
            mlflow_folder=self.mlflow_folder,
            cli_args=self.args,
            ckpt_config=self.config.get("checkpoint_saving", {}),
            # pass the checkpointing config if defined
        )

        # Define checkpointing callback for trainer
        config = self.config.get("checkpoint_saving")
        if config:
            checkpoint_callback = ModelCheckpoint(
                filename="checkpoint-{epoch}",
                every_n_epochs=config["every_n_epochs"],
                save_top_k=config["keep_last_n_ckpts"],
                monitor="epoch",  # monitor the metric "epoch" for selecting which checkpoints to save
                mode="max",  # get the max of the monitored metric
                save_last=config["save_last"],
                save_weights_only=config["save_weights_only"],
            )
            enable_checkpointing = True
        else:
            checkpoint_callback = None
            enable_checkpointing = False

        # Return trainer linked to callbacks and logger
        return lightning.Trainer(
            max_epochs=self.config["n_epochs"],
            accelerator=self.accelerator,
            logger=mlf_logger,
            enable_checkpointing=enable_checkpointing,
            callbacks=checkpoint_callback,
            fast_dev_run=self.fast_dev_run,
            limit_train_batches=self.limit_train_batches,
        )

    def train_model(self):
        # Create data module
        data_module = CrabsDataModule(
            self.images_dirs,
            self.annotation_files,
            self.config,
            self.seed_n,
        )

        if self.args.optuna:
            # Optimize hyperparameters
            best_hyperparameters = optimize_hyperparameters(
                self.config,
                data_module,
                self.accelerator,
                self.experiment_name,
                self.mlflow_folder,
                self.args,
                fast_dev_run=self.fast_dev_run,
                limit_train_batches=self.limit_train_batches,
            )

            # Update the config with the best hyperparameters
            self.config.update(best_hyperparameters)

        # Get model
        lightning_model = FasterRCNN(self.config)

        # Run training
        trainer = self.setup_trainer()
        trainer.fit(lightning_model, data_module)

        # if this is a slurm job: add slurm logs as artifacts
        slurm_job_id = os.environ.get("SLURM_JOB_ID")
        if slurm_job_id:
            slurm_logs_as_artifacts(trainer.logger, slurm_job_id)


def main(args) -> None:
    """
    Main function to orchestrate the training process.

    Parameters
    ----------
    args: argparse.Namespace
        An object containing the parsed command-line arguments.

    Returns
    -------
    None
    """
    trainer = DectectorTrain(args)
    trainer.train_model()


def train_parse_args(args):
    parser = argparse.ArgumentParser()
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
        help=(
            "list of paths to annotation files. The full path or the filename can be provided. "
            "If only filename is provided, it is assumed to be under dataset/annotations."
        ),
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
            "Accelerator for Pytorch Lightning. Valid inputs are: cpu, gpu, tpu, ipu, auto, mps. Default: gpu"
            "See https://lightning.ai/docs/pytorch/stable/common/trainer.html#accelerator "
            "and https://lightning.ai/docs/pytorch/stable/accelerators/mps_basic.html#run-on-apple-silicon-gpus"
        ),
    )
    parser.add_argument(
        "--experiment_name",
        type=str,
        default="Sept2023",
        help=(
            "Name of the experiment in MLflow, under which the current run will be logged. "
            "For example, the name of the dataset could be used, to group runs using the same data. "
            "Default: Sep2023"
        ),
    )
    parser.add_argument(
        "--seed_n",
        type=int,
        default=42,
        help="Seed for dataset splits. Default: 42",
    )
    parser.add_argument(
        "--fast_dev_run",
        action="store_true",
        help="Debugging option to run training for one batch and one epoch",
    )
    parser.add_argument(
        "--limit_train_batches",
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
        "--optuna",
        action="store_true",
        help="running optuna",
    )

    return parser.parse_args(args)


def app_wrapper():
    torch.set_float32_matmul_precision("medium")

    train_args = train_parse_args(sys.argv[1:])
    main(train_args)


if __name__ == "__main__":
    app_wrapper()
