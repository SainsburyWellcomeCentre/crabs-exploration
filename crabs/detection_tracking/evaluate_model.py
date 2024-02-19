import argparse
import os
import torch

from crabs.detection_tracking.detection_utils import load_dataset
from crabs.detection_tracking.evaluate import (
    compute_confusion_metrics,
    compute_precision_recall,
    save_images_with_boxes,
)

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

    def evaluate_model(self) -> None:
        """
        Evaluate the pre-trained model on the testation dataset.

        Returns:
            None
        """
        self._load_pretrain_model()
        self.trained_model.eval()
        evaluate_dataloader = load_dataset(
            self.main_dir, self.annotation, batch_size=1
        )

        # pdb.set_trace()
        with torch.no_grad():
            all_detections = []
            all_targets = []
            for imgs, annotations in evaluate_dataloader:
                imgs = list(img.to(device) for img in imgs)
                targets = [
                    {k: v.to(device) for k, v in t.items()}
                    for t in annotations
                ]
                detections = self.trained_model(imgs)

                all_detections.extend(detections)
                all_targets.extend(targets)

            class_stats = {"crab": {"tp": 0, "fp": 0, "fn": 0}}
            class_stats = compute_confusion_metrics(
                all_targets,  # one elem per image
                all_detections,
                self.ious_threshold,
                class_stats,
            )

            # Calculate precision, recall, and F1 score for each threshold
            compute_precision_recall(class_stats)

            save_images_with_boxes(
                evaluate_dataloader,
                self.trained_model,
                self.score_threshold,
                device,
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
