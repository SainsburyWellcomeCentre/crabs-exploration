import argparse
import json
import os

import torch
from detection_utils import (
    create_dataloader,
    get_test_transform,
    myFasterRCNNDataset,
)
from evaluate import test_detection

# select device (whether GPU or CPU)
device = (
    torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
)


class Detector_Test:
    """
    A class for testing object detection models using pre-trained classification.

    Args:
        args (argparse.Namespace): Command-line arguments containing
        configuration settings.

    Attributes:
        args (argparse.Namespace): The command-line arguments provided.
        main_dir (str): The main directory path.
        score_threshold (float): The confidence threshold for detection scores.
        sort_crab (Sort): An instance of the sorting algorithm used for tracking.
        trained_model: The pre-trained subject classification model.
        test_data (str): The path to the directory containing test images.
        test_label (str): The path to the test annotation JSON file.
        test_dataset: An instance of myFasterRCNNDataset for test data.
        test_dataloader: The DataLoader for the test dataset.

    Methods:
        _load_pretrain_model(self) -> None:
            Load the pre-trained subject classification model.

        _load_dataset(self) -> None:
            Load images and annotation file for testing.

        test_model(self) -> None:
            Test the pre-trained model on the test dataset.

    """

    def __init__(self, args: argparse.Namespace) -> None:
        self.args = args
        self.main_dir = args.main_dir
        self.score_threshold = args.score_threshold
        self.ious_threshold = args.ious_threshold
        # self.sort_crab = Sort()

    def _load_pretrain_model(self) -> None:
        """
        Load the pre-trained subject classification model.
        """
        # Load the pre-trained subject predictor
        # TODO: deal with different model
        self.trained_model = torch.load(
            self.args.model_dir, map_location=torch.device("cpu")
        )

    def _load_dataset(self) -> None:
        """Load images and annotation file for training"""

        self.annotation = (
            f"{self.main_dir}/annotations/VIA_JSON_combined_coco_gen.json"
        )
        # self.annotation = f"{self.main_dir}/labels/test.json"

        with open(self.annotation) as json_file:
            coco_data = json.load(json_file)

        self.test_file_paths = []
        for image_info in coco_data["images"]:
            image_id = image_info["id"]
            image_id -= 1
            image_file = image_info["file_name"]
            video_file = image_file.split("_")[1]

            if video_file == "09.08.2023-03-Left":
                continue

            # taking the first 40 frames as training data
            if image_id % 50 < 40:
                continue
            else:
                self.test_file_paths.append(image_file)

        self.test_dataset = myFasterRCNNDataset(
            self.main_dir,
            self.test_file_paths,
            self.annotation,
            transforms=get_test_transform(),
        )

        self.test_dataloader = create_dataloader(self.test_dataset, 1)

    def test_model(self) -> None:
        """
        Test the pre-trained model on the testation dataset.

        If 'sort' is False, object detection is tested via 'test_detection' function.
        If 'sort' is True, object tracking is tested via 'test_tracking' function.

        Returns:
            None
        """
        self._load_pretrain_model()
        self.trained_model.eval()
        self._load_dataset()

        if not self.args.sort:
            test_detection(
                self.test_dataloader,
                self.trained_model,
                self.score_threshold,
                self.ious_threshold,
            )


def main(args):
    test = Detector_Test(args)
    test.test_model()


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
        default=0.1,
        help="threshold for prediction score",
    )
    parser.add_argument(
        "--ious_threshold",
        type=float,
        default=0.5,
        help="threshold for IOU",
    )
    parser.add_argument(
        "--sort",
        type=bool,
        default=False,
        help="running sort as tracker",
    )

    args = parser.parse_args()
    main(args)
