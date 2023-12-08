import argparse
import json

import torch
import yaml  # type: ignore
from detection_utils import (
    create_dataloader,
    get_train_transform,
    myFasterRCNNDataset,
    save_model,
)
from models import create_faster_rcnn, train_faster_rcnn

# select GPU if available
device = (
    torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
)


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
            self.config = yaml.safe_load(f)  # type: dict
        print(self.config)

    def _load_dataset(self) -> None:
        """Load images and annotation file for training"""

        self.annotation = (
            f"{self.main_dir}/annotations/VIA_JSON_combined_coco_gen.json"
        )

        with open(self.annotation) as json_file:
            coco_data = json.load(json_file)

        self.train_file_paths = []
        for image_info in coco_data["images"]:
            image_id = image_info["id"]
            image_id -= 1
            image_file = image_info["file_name"]
            video_file = image_file.split("_")[1]

            if (
                video_file == "09.08.2023-03-Left"
                or video_file == "10.08.2023-01-Left"
                or video_file == "10.08.2023-01-Right"
            ):
                continue

            # taking the first 40 frames as training data
            if image_id % 50 < 40:
                self.train_file_paths.append(image_file)

        self.train_dataset = myFasterRCNNDataset(
            self.main_dir,
            self.train_file_paths,
            self.annotation,
            transforms=get_train_transform(),
        )

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

        trained_model = train_faster_rcnn(
            self.config, self.model, self.train_dataloader, optimizer
        )

        if self.config["save"]:
            save_model(trained_model)


def main(args) -> None:
    """
    Main function to orchestrate the training process.

    Args:
        args: Arguments or configuration settings.

    Returns:
        None
    """
    trainer = Dectector_Train(args)
    trainer.train_model()


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
    main(args)
