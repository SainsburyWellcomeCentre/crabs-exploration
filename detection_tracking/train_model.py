import argparse

import torch
import mlflow
import yaml
from pathlib import Path
from datetime import datetime

from _utils import create_dataloader, get_train_transform, save_model
from _models import create_faster_rcnn, train_faster_rcnn

# select device (whether GPU or CPU)
device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

# # Create experiment directory
# MODEL_REGISTRY = Path("experiments")
# Path(MODEL_REGISTRY).mkdir(exist_ok=True)
# # Set tracking URI
# mlflow.set_tracking_uri("file://" + str(MODEL_REGISTRY.absolute()))
# # set experiment
# mlflow.set_experiment(experiment_name="baseline")

# EXPERIMENT_NAME = "baseline"
# RUN_NAME = f"run_{datetime.today()}"
# print(EXPERIMENT_NAME)

# try:
#     EXPERIMENT_ID = mlflow.get_experiment_by_name(EXPERIMENT_NAME).experiment_id
#     print(EXPERIMENT_ID)
# except:
#     EXPERIMENT_ID = mlflow.create_experiment(EXPERIMENT_NAME)
#     print(EXPERIMENT_ID)


class Dectector_Train:
    """Training class for detector algorithm

    Args
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

    def __init__(self, args: argparse.Namespace) -> None:
        self.config_file = args.config_file
        self.main_dir = args.main_dir
        self.model_name = args.model_name
        self.load_config_yaml()

    def load_config_yaml(self) -> None:
        """Load a YAML file describing the training setup"""

        with open(self.config_file, "r") as f:
            self.config = yaml.safe_load(f)

    def _load_dataset(self) -> None:
        """Load images and annotation file for training"""

        self.train_data = f"{self.main_dir}/images/train/"
        self.train_label = f"{self.main_dir}/labels/train.json"

        if self.model_name == "faster_rcnn":
            from _utils import myFasterRCNNDataset

            self.train_dataset = myFasterRCNNDataset(
                self.train_data, self.train_label, transforms=get_train_transform()
            )

        print(self.config["batch_size"])

        self.train_dataloader = create_dataloader(
            self.train_dataset, self.config["batch_size"]
        )

    def train_model(self) -> None:
        """Train the model using the provided configuration"""

        self._load_dataset()

        # create model
        self.model = create_faster_rcnn(
            num_classes=self.config["num_classes"], coco_model=True
        )
        self.model.to(device)

        optimizer = torch.optim.Adam(
            self.model.parameters(),
            lr=self.config["learning_rate"],
            weight_decay=self.config["wdecay"],
        )

        print(f"Training {self.model_name}...")

        trained_model = train_faster_rcnn(
            self.config, self.model, self.train_dataloader, optimizer
        )

        if self.config["save"]:
            save_model(trained_model)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config_file",
        type=str,
        default="config/faster_rcnn.yaml",
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
        help="the model to use to train the object detection. Options: faster_rcnn",
    )
    args = parser.parse_args()

    print(args)
    trainer = Dectector_Train(args)
    trainer.train_model()
