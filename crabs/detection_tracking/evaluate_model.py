import argparse
import json
import os

import torch
from detection_utils import (
    create_dataloader,
    get_test_transform,
    myFasterRCNNDataset,
)
from evaluate import evaluate_detection

# select device (whether GPU or CPU)
device = (
    torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
)


class Detector_Evaluate:
    """
    A class for evaluating object detection models using pre-trained classification.

    Parameters
    ----------
    args : argparse
        Command-line arguments containing configuration settings.

    Attributes
    ----------
    args : argparse
        The command-line arguments provided.
    main_dir : str
        The main directory path.
    annotation_file : str
        The filename of coco annotation JSON file.
    score_threshold : float
        The score threshold for confidence detection.
    ious_threshold : float
        The ious threshold for detection bounding boxes.
    trained_model:
        The pre-trained subject classification model.
    evaluate_dataset:
        An instance of myFasterRCNNDataset for test data.
    evaluate_dataloader:
        The DataLoader for the test dataset.
    """

    def __init__(self, args: argparse.Namespace) -> None:
        self.args = args
        self.main_dir = args.main_dir
        self.annotation_file = args.annotation_file
        self.ious_threshold = args.ious_threshold
        self.score_threshold = args.score_threshold
        self.annotation = f"{self.main_dir}/annotations/{self.annotation_file}"

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

        with open(self.annotation) as json_file:
            coco_data = json.load(json_file)

        self.evaluate_file_paths = []
        for image_info in coco_data["images"]:
            image_id = image_info["id"]
            image_id -= 1  # reset the image_id to 0 to get the index
            image_file = image_info["file_name"]
            video_file = image_file.split("_")[1]

            if video_file == "09.08.2023-03-Left":
                continue

            # taking the first 40 frames per video as training data
            if image_id % 50 < 40:
                continue
            else:
                self.evaluate_file_paths.append(image_file)

        self.evaluate_dataset = myFasterRCNNDataset(
            self.main_dir,
            self.evaluate_file_paths,
            self.annotation,
            transforms=get_test_transform(),
        )

        self.evaluate_dataloader = create_dataloader(self.evaluate_dataset, 1)

    def evaluate_model(self) -> None:
        """
        Evaluate the pre-trained model on the testation dataset.

        Returns:
            None
        """
        self._load_pretrain_model()
        self.trained_model.eval()
        self._load_dataset()

        # pdb.set_trace()
        evaluate_detection(
            self.evaluate_dataloader,
            self.trained_model,
            self.ious_threshold,
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
    eval = Detector_Evaluate(args)
    eval.evaluate_model()


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
        help="main location of images and coco annotation",
    )
    parser.add_argument(
        "--annotation_file",
        type=str,
        required=True,
        help="filename for coco annotation",
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
        help="threshold for confidence score",
    )
    parser.add_argument(
        "--ious_threshold",
        type=float,
        default=0.5,
        help="threshold for IOU",
    )

    args = parser.parse_args()
    main(args)
