import argparse
import logging

import torch
import torchvision

from crabs.detection.dataloaders import CustomDataLoader
from crabs.detection.visualisation import save_images_with_boxes

logging.basicConfig(level=logging.INFO)


def compute_precision_recall(class_stats):
    """
    Compute precision and recall.

    Parameters
    ----------
    class_stats : dict
        Statistics or information about different classes.

    Returns
    ----------
    None
    """

    for _, stats in class_stats.items():
        precision = stats["tp"] / max(stats["tp"] + stats["fp"], 1)
        recall = stats["tp"] / max(stats["tp"] + stats["fn"], 1)

        logging.info(
            f"Precision: {precision:.4f}, Recall: {recall:.4f}, "
            f"False Positive: {class_stats['crab']['fp']}, "
            f"False Negative: {class_stats['crab']['fn']}"
        )


def compute_confusion_metrics(targets, detections, ious_threshold) -> None:
    """
    Compute metrics (true positive, false positive, false negative) for object detection.

    Parameters
    ----------
    targets : list
        Ground truth annotations.
    detections : list
        Detected objects.
    ious_threshold  : float
        The threshold value for the intersection-over-union (IOU).
        Only detections whose IOU relative to the ground truth is above the
        threshold are true positive candidates.
    class_stats : dict
        Statistics or information about different classes.

    Returns
    ----------
    None
    """
    class_stats = {"crab": {"tp": 0, "fp": 0, "fn": 0}}
    for target, detection in zip(targets, detections):
        gt_boxes = target["boxes"]
        pred_boxes = detection["boxes"]
        pred_labels = detection["labels"]

        ious = torchvision.ops.box_iou(pred_boxes, gt_boxes)

        max_ious, max_indices = ious.max(dim=1)

        # Identify true positives, false positives, and false negatives
        for idx, iou in enumerate(max_ious):
            if iou.item() > ious_threshold:
                pred_class_idx = pred_labels[idx].item()
                true_label = target["labels"][max_indices[idx]].item()

                if pred_class_idx == true_label:
                    class_stats["crab"]["tp"] += 1
                else:
                    class_stats["crab"]["fp"] += 1
            else:
                class_stats["crab"]["fp"] += 1

        for target_box_index, target_box in enumerate(gt_boxes):
            found_match = False
            for idx, iou in enumerate(max_ious):
                if (
                    iou.item()
                    > ious_threshold  # we need this condition because the max overlap is not necessarily above the threshold
                    and max_indices[idx]
                    == target_box_index  # the matching index is the index of the GT box with which it has max overlap
                ):
                    # There's an IoU match and the matched index corresponds to the current target_box_index
                    found_match = True
                    break  # Exit loop, a match was found

            if not found_match:
                # print(found_match)
                class_stats["crab"][
                    "fn"
                ] += 1  # Ground truth box has no corresponding detection

    compute_precision_recall(class_stats)


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
        Load the pre-trained subject classification model.

        Returns
        -------
        None
        """
        self.trained_model = torch.load(
            self.args.model_dir, map_location=torch.device("cpu")
        )

    def evaluate_model(self) -> None:
        """
        Evaluate the pre-trained model on the testation dataset.

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

    data_module = CustomDataLoader(main_dir, annotation, batch_size=1)
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
