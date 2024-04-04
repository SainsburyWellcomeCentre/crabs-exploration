import argparse

import torch
import yaml  # type: ignore

from crabs.detection_tracking.datamodules import CrabsDataModule
from crabs.detection_tracking.evaluate import (
    compute_confusion_matrix_elements,
    save_images_with_boxes,
)


class DetectorEvaluation:
    """
    A class for evaluating an object detector using trained model.

    Parameters
    ----------
    args : argparse
        Command-line arguments containing configuration settings.
    score_threshold : float
        The score threshold for confidence detection.
    ious_threshold : float
        The ious threshold for detection bounding boxes.
    evaluate_dataloader:
        The DataLoader for the test dataset.
    """

    def __init__(
        self,
        args: argparse.Namespace,
        data_loader: torch.utils.data.DataLoader,
    ) -> None:
        self.args = args
        self.ious_threshold = args.ious_threshold
        self.score_threshold = args.score_threshold
        self.evaluate_dataloader = data_loader

    def _load_trained_model(self) -> None:
        """
        Load the trained model.

        Returns
        -------
        None
        """
        self.trained_model = torch.load(
            self.args.model_dir,
            map_location=torch.device(self.args.accelerator),
        )

    def evaluate_model(self) -> None:
        """
        Evaluate the trained model on the test dataset.

        Returns
        -------
        None
        """
        self._load_trained_model()

        # set model in eval mode
        self.trained_model.eval()

        # select device
        device = (
            torch.device("cuda")
            if torch.cuda.is_available()
            else torch.device("cpu")
        )

        all_detections = []
        all_targets = []

        with torch.no_grad():
            for imgs, annotations in self.evaluate_dataloader:
                imgs = list(img.to(device) for img in imgs)
                targets = [
                    {k: v.to(device) for k, v in t.items() if k != "image_id"}
                    for t in annotations
                ]
                detections = self.trained_model(imgs)

                all_detections.extend(detections)
                all_targets.extend(targets)

        compute_confusion_matrix_elements(
            all_targets,  # one elem per image
            all_detections,
            self.ious_threshold,
        )

        if self.args.save_frames:
            save_images_with_boxes(
                self.evaluate_dataloader,
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
    list_images_dirs = args.images_dirs

    # get annotations
    list_annotations_files = args.annotation_files
    # get config
    with open(args.config_file, "r") as f:
        config = yaml.safe_load(f)

    # get dataloader
    data_module = CrabsDataModule(
        list_images_dirs,
        list_annotations_files,
        config,
        args.seed_n,
    )
    data_module.setup()
    data_loader = data_module.test_dataloader()

    # evaluator
    evaluator = DetectorEvaluation(args, data_loader)
    evaluator.evaluate_model()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config_file",
        type=str,
        default="crabs/detection_tracking/config/faster_rcnn.yaml",
        help="location of YAML config to control training",
    )
    parser.add_argument(
        "--model_dir",
        type=str,
        required=True,
        help="location of trained model",
    )
    parser.add_argument(
        "--images_dirs",
        type=str,
        nargs="+",
        required=True,
        help="list of paths to images directories",
    )
    parser.add_argument(
        "--annotation_files",
        type=str,
        nargs="+",
        required=True,
        help="list of paths to annotation files",
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
    parser.add_argument(
        "--accelerator",
        type=str,
        default="gpu",
        help="accelerator for pytorch lightning",
    )
    parser.add_argument(
        "--seed_n",
        type=int,
        default=42,
        help="seed for random state",
    )
    parser.add_argument(
        "--save_frames",
        action="store_true",
        help=("Save predicted frames with bboxes."),
    )

    args = parser.parse_args()
    main(args)
