import torch
import torchvision
import argparse
import os
import cv2
from PIL import Image
import torchvision.transforms as transforms
import numpy as np
from _utils import create_dataloader, get_test_transform
from _utils import myFasterRCNNDataset
from sort import *
import numpy as np


# select device (whether GPU or CPU)
# device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
device = "cpu"


class Detector_Test:
    def __init__(self, args: argparse.Namespace) -> None:
        self.args = args
        self.main_dir = args.main_dir
        self.score_threshold = args.score_threshold

    def _load_pretrain_model(self) -> None:
        """Load the pretrain subject classification model"""

        # Load the pretrain subject predictor
        # TODO:deal with different model
        self.trained_model = torch.load(
            self.args.model_dir,
            map_location=torch.device('cpu')

        )

    def _load_dataset(self) -> None:
        """Load images and annotation file for training"""

        self.valid_data = f"{self.main_dir}/images/test/"
        self.valid_label = f"{self.main_dir}/labels/test.json"

        self.valid_dataset = myFasterRCNNDataset(
            self.valid_data, self.valid_label, transforms=get_test_transform()
        )

        self.valid_dataloader = create_dataloader(self.valid_dataset, 1)

    def test_model(self) -> None:
        self._load_pretrain_model()
        self.trained_model.eval()
        self._load_dataset()

        if not self.args.sort:
            from _test import test_detection

            test_detection(
                self.valid_dataloader, self.trained_model, self.score_threshold
            )
        else:
            from _test import test_tracking

            test_tracking(
                self.valid_dataloader, self.trained_model, self.score_threshold
            )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model_dir",
        type=str,
        required=True,
        help="location of trained model",
    )
    parser.add_argument(
        "--main_dir",
        type=str,
        required=True,
        help="location of images and coco annotation",
    )
    parser.add_argument(
        "--save",
        type=bool,
        default=True,
        help="save video inference",
    )
    parser.add_argument(
        "--output_path",
        type=str,
        default=os.getcwd(),
        help="location of output video",
    )
    parser.add_argument(
        "--score_threshold",
        type=float,
        default=0.5,
        help="threshold for prediction score",
    )
    parser.add_argument(
        "--sort",
        type=bool,
        default=False,
        help="running sort as tracker",
    )

    args = parser.parse_args()
    test = Detector_Test(args)
    test.test_model()
