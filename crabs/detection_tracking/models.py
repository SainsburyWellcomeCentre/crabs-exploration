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
        self.model = torchvision.models.detection.fasterrcnn_resnet50_fpn_v2(
            weights=torchvision.models.detection.FasterRCNN_ResNet50_FPN_V2_Weights.DEFAULT
        )
        in_features = self.model.roi_heads.box_predictor.cls_score.in_features
        self.model.roi_heads.box_predictor = FastRCNNPredictor(
            in_features, config["num_classes"]
        )
        self.training_step_outputs = {
            "total_training_loss": 0.0,
            "num_batches": 0,
        }
        self.validation_step_outputs = {
            "total_precision": 0.0,
            "num_batches": 0,
        }

    def forward(self, x):
        return self.model(x)

    def training_step(self, batch, batch_idx):
        images, targets = batch
        loss_dict = self.model(images, targets)
        total_loss = sum(loss for loss in loss_dict.values())

        # Accumulate the loss over each step during the epoch
        self.training_step_outputs["total_training_loss"] += total_loss.item()
        self.training_step_outputs["num_batches"] += 1

        return total_loss

    def on_train_epoch_end(self):
        avg_loss = (
            self.training_step_outputs["total_training_loss"]
            / self.training_step_outputs["num_batches"]
        )
        self.logger.log_metrics(
            {"train_loss": avg_loss}, step=self.current_epoch
        )
        # Reset the training_step_outputs for the next epoch
        self.training_step_outputs = {
            "total_training_loss": 0.0,
            "num_batches": 0,
        }

    def validation_step(self, batch, batch_idx):
        images, targets = batch
        predictions = self.model(images, targets)
        (
            precision,
            _,
            _,
        ) = compute_confusion_matrix_elements(
            targets, predictions, self.config["iou_threshold"]
        )
        self.validation_step_outputs["total_precision"] += precision
        self.validation_step_outputs["num_batches"] += 1
        return precision

    def on_validation_epoch_end(self):
        # Compute the mean precision across all batches in the epoch
        mean_precision = (
            self.validation_step_outputs["total_precision"]
            / self.validation_step_outputs["num_batches"]
        )

        self.logger.log_metrics(
            {"val_precision": mean_precision}, step=self.current_epoch
        )
        self.validation_step_outputs = {
            "total_precision": 0.0,
            "num_batches": 0,
        }

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(
            self.parameters(),
            lr=self.config["learning_rate"],
            weight_decay=self.config["wdecay"],
        )
        return {
            "optimizer": optimizer,
        }