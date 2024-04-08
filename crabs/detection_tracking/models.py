import logging

import torch
import torchvision
from lightning import LightningModule
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor

from crabs.detection_tracking.evaluate import compute_confusion_matrix_elements


class FasterRCNN(LightningModule):
    """
    LightningModule implementation of Faster R-CNN for object detection.
    Parameters:
    -----------
    config : dict
        Configuration settings for the model.
    Methods:
    --------
    forward(x):
        Forward pass of the model.
    training_step(batch, batch_idx):
        Defines the training step for the model.
    configure_optimizers():
        Configures the optimizer for training.
    Attributes:
    -----------
    config : dict
        Configuration settings for the model.
    model : torch.nn.Module
        Faster R-CNN model.
    """
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.model = self.configure_model()
        self.training_step_outputs = {
            "total_training_loss": 0.0,
            "num_batches": 0,
        }
        self.validation_step_outputs = {
            "total_precision": 0.0,
            "total_recall": 0.0,
            "num_batches": 0,
        }
        self.test_step_outputs = {
            "total_precision": 0.0,
            "total_recall": 0.0,
            "num_batches": 0,
        }

    def configure_model(self):
        model = torchvision.models.detection.fasterrcnn_resnet50_fpn_v2(
            weights=torchvision.models.detection.FasterRCNN_ResNet50_FPN_V2_Weights.DEFAULT
        )
        in_features = model.roi_heads.box_predictor.cls_score.in_features
        model.roi_heads.box_predictor = FastRCNNPredictor(
            in_features, self.config["num_classes"]
        )
        return model

    def forward(self, x):
        return self.model(x)

    def accumulate_metrics(self, outputs, step_outputs):
        for output in outputs:
            step_outputs["total_precision"] += output["precision"]
            step_outputs["total_recall"] += output["recall"]
            step_outputs["num_batches"] += 1

    def on_train_epoch_end(self):
        avg_loss = (
            self.training_step_outputs["total_training_loss"]
            / self.training_step_outputs["num_batches"]
        )
        self.logger.log_metrics(
            {"train_loss": avg_loss}, step=self.current_epoch
        )
        self.training_step_outputs = {
            "total_training_loss": 0.0,
            "num_batches": 0,
        }

    def on_validation_epoch_end(self):
        mean_precision = (
            self.validation_step_outputs["total_precision"]
            / self.validation_step_outputs["num_batches"]
        )
        mean_recall = (
            self.validation_step_outputs["total_recall"]
            / self.validation_step_outputs["num_batches"]
        )
        logging.info(
            f"Average Precision: {mean_precision:.4f}, Average Recall: {mean_recall:.4f}"
        )
        self.logger.log_metrics(
            {"val_precision": mean_precision}, step=self.current_epoch
        )
        self.logger.log_metrics(
            {"val_recall": mean_recall}, step=self.current_epoch
        )
        self.validation_step_outputs = {
            "total_precision": 0.0,
            "total_recall": 0.0,
            "num_batches": 0,
        }

    def on_test_epoch_end(self):
        mean_precision = (
            self.test_step_outputs["total_precision"]
            / self.test_step_outputs["num_batches"]
        )
        mean_recall = (
            self.test_step_outputs["total_recall"]
            / self.test_step_outputs["num_batches"]
        )
        logging.info(
            f"Average Precision: {mean_precision:.4f}, Average Recall: {mean_recall:.4f}"
        )
        self.logger.log_metrics(
            {"test_avg_precision": mean_precision}, step=self.current_epoch
        )
        self.logger.log_metrics(
            {"test_avg_recall": mean_recall}, step=self.current_epoch
        )
        self.test_step_outputs = {
            "total_precision": 0.0,
            "total_recall": 0.0,
            "num_batches": 0,
        }

    def training_step(self, batch, batch_idx):
        images, targets = batch
        loss_dict = self.model(images, targets)
        total_loss = sum(loss for loss in loss_dict.values())
        self.training_step_outputs["total_training_loss"] += total_loss.item()
        self.training_step_outputs["num_batches"] += 1
        return total_loss

    def shared_step(self, batch):
        images, targets = batch
        predictions = self.model(images)
        precision, recall, _ = compute_confusion_matrix_elements(
            targets, predictions, self.config["iou_threshold"]
        )
        return {"precision": precision, "recall": recall}

    def validation_step(self, batch, batch_idx):
        outputs = self.shared_step(batch)
        self.accumulate_metrics([outputs], self.validation_step_outputs)
        return outputs

    def test_step(self, batch, batch_idx):
        outputs = self.shared_step(batch)
        self.accumulate_metrics([outputs], self.test_step_outputs)
        return outputs

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(
            self.parameters(),
            lr=self.config["learning_rate"],
            weight_decay=self.config["wdecay"],
        )
        return {"optimizer": optimizer}
