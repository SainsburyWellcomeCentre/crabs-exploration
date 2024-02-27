import argparse

import torch

from crabs.detection_tracking.datamodule import myDataModule
from crabs.detection_tracking.evaluate import (
    compute_confusion_metrics,
    save_images_with_boxes,
)


class Detector_Evaluation:
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

    def _load_pretrain_model(self) -> None:
        """
        Load the trained model.

        Returns
        -------
        None
        """
        self.trained_model = torch.load(
            self.args.model_dir, map_location=torch.device("cpu")
        )

    def evaluate_model(self) -> None:
        """
        Evaluate the trained model on the test dataset.

        Returns
        -------
        None
        """
        self._load_pretrain_model()
        self.trained_model.eval()
        device = (
            torch.device("cuda")
            if torch.cuda.is_available()
            else torch.device("cpu")
        )

        with torch.no_grad():
            all_detections = []
            all_targets = []
            for imgs, annotations in self.evaluate_dataloader:
                imgs = list(img.to(device) for img in imgs)
                targets = [
                    {k: v.to(device) for k, v in t.items()}
                    for t in annotations
                ]
                detections = self.trained_model(imgs)

                all_detections.extend(detections)
                all_targets.extend(targets)

            compute_confusion_metrics(
                all_targets,  # one elem per image
                all_detections,
                self.ious_threshold,
            )

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

    main_dir = args.main_dir
    annotation_file = args.annotation_file
    annotation = f"{main_dir}/annotations/{annotation_file}"

    data_module = myDataModule(main_dir, annotation, batch_size=1)
    data_module.setup()
    data_loader = data_module.val_dataloader()

    evaluator = Detector_Evaluation(args, data_loader)
    evaluator.evaluate_model()


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
        default="cpu",
        help="accelerator for pytorch lightning",
    )

    args = parser.parse_args()
    main(args)
