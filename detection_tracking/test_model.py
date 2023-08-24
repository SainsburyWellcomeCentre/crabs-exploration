import argparse
import os

import torch
from _utils import create_dataloader, get_test_transform, myFasterRCNNDataset
from sort import Sort

# select device (whether GPU or CPU)
device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")


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
        self.sort_crab = Sort()

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
        """
        Load images and annotation file for testing.
        """
        self.test_data = f"{self.main_dir}/images/test/"
        self.test_label = f"{self.main_dir}/labels/test.json"

        self.test_dataset = myFasterRCNNDataset(
            self.test_data, self.test_label, transforms=get_test_transform()
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
            from _test import test_detection

            test_detection(
                self.test_dataloader, self.trained_model, self.score_threshold
            )
        else:
            from _test import test_tracking

            test_tracking(
                self.test_dataloader,
                self.trained_model,
                self.score_threshold,
                self.sort_crab,
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
