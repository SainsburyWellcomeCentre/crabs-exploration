"""LightningModule for Faster R-CNN for object detection."""

import logging
from typing import Any, Union

import torch
from lightning import LightningModule
from torchvision.models.detection import (
    faster_rcnn,
    fasterrcnn_resnet50_fpn_v2,
)

from crabs.detector.utils.evaluate import (
    compute_precision_recall,
)


class FasterRCNN(LightningModule):
    """LightningModule implementation of Faster R-CNN for object detection.

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

    def __init__(self, config: dict[str, Any], optuna_log=False):
        """Initialise the Faster R-CNN model with the given configuration."""
        super().__init__()
        self.config = config
        self.model = self.configure_model()
        self.optuna_log = optuna_log

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
        """Configure Faster R-CNN model.

        Use default weights,
        specified backbone, and box predictor.
        """
        model = fasterrcnn_resnet50_fpn_v2(weights="DEFAULT")
        in_features = model.roi_heads.box_predictor.cls_score.in_features
        model.roi_heads.box_predictor = faster_rcnn.FastRCNNPredictor(
            in_features, self.config["num_classes"]
        )
        return model

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass of the model."""
        return self.model(x)

    def accumulate_epoch_metrics(
        self,
        batch_output: dict,
        dataset_str: str,
    ) -> None:
        """Accumulates precision and recall metrics per epoch."""
        getattr(self, f"{dataset_str}_step_outputs")["precision_epoch"] += (
            batch_output["precision"]
        )

        getattr(self, f"{dataset_str}_step_outputs")["recall_epoch"] += (
            batch_output["recall"]
        )

        getattr(self, f"{dataset_str}_step_outputs")["num_batches"] += 1

    def compute_precision_recall_epoch(
        self, step_outputs: dict[str, Union[float, int]], log_str: str
    ) -> tuple[float, float]:
        """Compute and log mean precision and recall for the current epoch."""
        # compute mean precision and recall
        mean_precision = (
            step_outputs["precision_epoch"] / step_outputs["num_batches"]
        )
        mean_recall = (
            step_outputs["recall_epoch"] / step_outputs["num_batches"]
        )

        # add metrics to logger
        self.logger.log_metrics(
            {f"{log_str}_precision": mean_precision}, step=self.current_epoch
        )
        self.logger.log_metrics(
            {f"{log_str}_recall": mean_recall}, step=self.current_epoch
        )

        # log to screen
        logging.info(
            f"Average Precision ({log_str}): {mean_precision:.4f},"
            f"Average Recall ({log_str}): {mean_recall:.4f}"
        )

        return mean_precision, mean_recall

    def on_train_epoch_end(self) -> None:
        """Define hook called after each training epoch.

        Used to perform tasks such as logging and resetting metrics.
        """
        # compute average loss
        avg_loss = (
            self.training_step_outputs["training_loss_epoch"]
            / self.training_step_outputs["num_batches"]
        )

        # log
        self.logger.log_metrics(
            {"train_loss": avg_loss}, step=self.current_epoch
        )

        # reset
        self.training_step_outputs = {
            "training_loss_epoch": 0.0,
            "num_batches": 0,
        }

    def on_validation_epoch_end(self) -> None:
        """Define hook called after each validation epoch.

        Used to compute metrics and logging.
        """
        (val_precision, val_recall) = self.compute_precision_recall_epoch(
            self.validation_step_outputs, "val"
        )

        # we need these logs for hyperparameter optimisation
        if self.optuna_log:
            self.log("val_precision_optuna", val_precision)
            self.log("val_recall_optuna", val_recall)

        # Reset metrics for next epoch
        self.validation_step_outputs = {
            "precision_epoch": 0.0,
            "recall_epoch": 0.0,
            "num_batches": 0,
        }

    def on_test_epoch_end(self) -> None:
        """Define hook called after each testing epoch.

        Used to compute metrics and logging.
        """
        (test_precision, test_recall) = self.compute_precision_recall_epoch(
            self.test_step_outputs, "test"
        )

        # Reset metrics for next epoch
        self.test_step_outputs = {
            "precision_epoch": 0.0,
            "recall_epoch": 0.0,
            "num_batches": 0,
        }

    def training_step(
        self, batch: tuple[torch.Tensor, torch.Tensor], batch_idx: int
    ) -> torch.Tensor:
        """Define training step for the model."""
        images, targets = batch
        loss_dict = self.model(images, targets)
        total_loss = sum(loss for loss in loss_dict.values())

        self.training_step_outputs["training_loss_epoch"] += total_loss.item()
        self.training_step_outputs["num_batches"] += 1

        return total_loss

    def val_test_step(
        self, batch: tuple[torch.Tensor, torch.Tensor]
    ) -> dict[str, Union[float, int]]:
        """Perform inference on a validation or test batch.

        Computes precision and recall.
        """
        images, targets = batch
        predictions = self.model(images)

        precision, recall = compute_precision_recall(
            predictions, targets, self.config["iou_threshold"]
        )

        return {"precision": precision, "recall": recall}

    def validation_step(
        self, batch: tuple[torch.Tensor, torch.Tensor], batch_idx: int
    ) -> dict[str, Union[float, int]]:
        """Define the validation step for the model."""
        outputs = self.val_test_step(batch)

        # log to screen
        logging.info(
            f"batch {batch_idx} Precision: {outputs['precision']:.4f},"
            f"batch {batch_idx} Recall: {outputs['recall']:.4f}"
        )

        self.accumulate_epoch_metrics(outputs, "validation")
        return outputs

    def test_step(
        self, batch: tuple[torch.Tensor, torch.Tensor], batch_idx: int
    ) -> dict[str, Union[float, int]]:
        """Define the test step for the model."""
        outputs = self.val_test_step(batch)
        self.accumulate_epoch_metrics(outputs, "test")
        return outputs

    def configure_optimizers(self) -> dict[str, torch.optim.Optimizer]:
        """Configure the optimizer for training."""
        optimizer = torch.optim.Adam(
            self.parameters(),
            lr=self.config["learning_rate"],
            weight_decay=self.config["wdecay"],
        )
        return {"optimizer": optimizer}
