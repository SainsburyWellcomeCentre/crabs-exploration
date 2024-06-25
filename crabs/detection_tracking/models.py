import logging
from typing import Any, Tuple, Union

import torch
from lightning import LightningModule
from torchvision.models.detection import (
    faster_rcnn,
    fasterrcnn_resnet50_fpn_v2,
)

from crabs.detection_tracking.evaluate_utils import (
    compute_confusion_matrix_elements,
)


class FasterRCNN(LightningModule):
    """
    LightningModule implementation of Faster R-CNN for object detection.

    Parameters
    ----------
    config : dict
        Configuration settings for the model.

    Methods
    -------
    forward(x):
        Forward pass of the model.
    training_step(batch, batch_idx):
        Defines the training step for the model.
    configure_optimizers():
        Configures the optimizer for training.

    Attributes
    ----------
    config : dict
        Configuration settings for the model.
    model : torch.nn.Module
        Faster R-CNN model.
    training_step_outputs : dict
        Dictionary to store training metrics.
    validation_step_outputs : dict
        Dictionary to store validation metrics.
    test_step_outputs : dict
        Dictionary to store test metrics.
    """

    def __init__(self, config: dict[str, Any]):
        super().__init__()
        self.config = config
        self.model = self.configure_model()

        # save all arguments passed to __init__
        self.save_hyperparameters()

        # metrics to log during training/val/test loop
        self.training_step_outputs = {
            "training_loss_epoch": 0.0,
            "num_batches": 0,
        }
        self.validation_step_outputs = {
            "precision_epoch": 0.0,
            "recall_epoch": 0.0,
            "num_batches": 0,
        }
        self.test_step_outputs = {
            "precision_epoch": 0.0,
            "recall_epoch": 0.0,
            "num_batches": 0,
        }

    def configure_model(self) -> torch.nn.Module:
        """
        Configures the Faster R-CNN model with default weights,
        specified backbone, and box predictor.
        """
        model = fasterrcnn_resnet50_fpn_v2(weights="DEFAULT")
        in_features = model.roi_heads.box_predictor.cls_score.in_features
        model.roi_heads.box_predictor = faster_rcnn.FastRCNNPredictor(
            in_features, self.config["num_classes"]
        )
        return model

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass of the model.
        """
        return self.model(x)

    def accumulate_epoch_metrics(
        self,
        batch_output: dict,
        dataset_str: str,
    ) -> None:
        """
        Accumulates precision and recall metrics per epoch.
        """
        getattr(self, f"{dataset_str}_step_outputs")[
            "precision_epoch"
        ] += batch_output["precision"]

        getattr(self, f"{dataset_str}_step_outputs")[
            "recall_epoch"
        ] += batch_output["recall"]

        getattr(self, f"{dataset_str}_step_outputs")["num_batches"] += 1

    def compute_precision_recall_epoch(
        self, step_outputs: dict[str, Union[float, int]], log_str: str
    ) -> Tuple[dict[str, Union[float, int]], float, float]:
        """
        Computes and logs mean precision and recall for the current epoch.
        """
        mean_precision = (
            step_outputs["precision_epoch"] / step_outputs["num_batches"]
        )
        mean_recall = (
            step_outputs["recall_epoch"] / step_outputs["num_batches"]
        )

        self.logger.log_metrics(
            {f"{log_str}_precision": mean_precision}, step=self.current_epoch
        )
        self.logger.log_metrics(
            {f"{log_str}_recall": mean_recall}, step=self.current_epoch
        )

        # Reset metrics for next epoch
        step_outputs = {
            "precision_epoch": 0.0,
            "recall_epoch": 0.0,
            "num_batches": 0,
        }

        return step_outputs, mean_precision, mean_recall

    def on_train_epoch_end(self) -> None:
        """
        Hook called after each training epoch to perform tasks such as logging and resetting metrics.
        """
        avg_loss = (
            self.training_step_outputs["training_loss_epoch"]
            / self.training_step_outputs["num_batches"]
        )
        self.logger.log_metrics(
            {"train_loss": avg_loss}, step=self.current_epoch
        )
        self.training_step_outputs = {
            "training_loss_epoch": 0.0,
            "num_batches": 0,
        }

    def on_validation_epoch_end(self) -> None:
        """
        Hook called after each validation epoch to compute metrics and logging.
        """
        (
            self.validation_step_outputs,
            val_precision,
            val_recall,
        ) = self.compute_precision_recall_epoch(
            self.validation_step_outputs, "val"
        )

        # we need these logs for hyperparameter optimisation
        self.log("val_precision", val_precision)
        self.log("val_recall", val_recall)

    def on_test_epoch_end(self) -> None:
        """
        Hook called after each testing epoch to compute metrics and logging.
        """
        (
            self.test_step_outputs,
            test_precision,
            test_recall,
        ) = self.compute_precision_recall_epoch(self.test_step_outputs, "test")

        logging.info(
            f"Test Average Precision: {test_precision:.4f},"
            f"Test Average Recall: {test_recall:.4f}"
        )

    def training_step(
        self, batch: tuple[torch.Tensor, torch.Tensor], batch_idx: int
    ) -> torch.Tensor:
        """
        Defines the training step for the model.
        """
        images, targets = batch
        loss_dict = self.model(images, targets)
        total_loss = sum(loss for loss in loss_dict.values())
        self.training_step_outputs["training_loss_epoch"] += total_loss.item()
        self.training_step_outputs["num_batches"] += 1
        return total_loss

    def val_test_step(
        self, batch: tuple[torch.Tensor, torch.Tensor]
    ) -> dict[str, Union[float, int]]:
        """
        Performs inference on a validation or test batch and computes precision and recall.
        """
        images, targets = batch
        predictions = self.model(images)

        precision, recall, _ = compute_confusion_matrix_elements(
            targets, predictions, self.config["iou_threshold"]
        )
        return {"precision": precision, "recall": recall}

    def validation_step(
        self, batch: tuple[torch.Tensor, torch.Tensor], batch_idx: int
    ) -> dict[str, Union[float, int]]:
        """
        Defines the validation step for the model.
        """
        outputs = self.val_test_step(batch)
        self.accumulate_epoch_metrics(outputs, "validation")
        return outputs

    def test_step(
        self, batch: tuple[torch.Tensor, torch.Tensor], batch_idx: int
    ) -> dict[str, Union[float, int]]:
        """
        Defines the test step for the model.
        """
        outputs = self.val_test_step(batch)
        self.accumulate_epoch_metrics(outputs, "test")
        return outputs

    def configure_optimizers(self) -> dict[str, torch.optim.Optimizer]:
        """
        Configures the optimizer for training.
        """
        optimizer = torch.optim.Adam(
            self.parameters(),
            lr=self.config["learning_rate"],
            weight_decay=self.config["wdecay"],
        )
        return {"optimizer": optimizer}
