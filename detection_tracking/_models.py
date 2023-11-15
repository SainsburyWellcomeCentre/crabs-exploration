import os
import pickle
import tempfile
import time
from datetime import datetime

import mlflow
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor


def create_faster_rcnn(num_classes: int, coco_model: bool = True) -> nn.Module:
    """Create faster rcnn model for the training

    Parameters
    ----------
    num_classes : int
        number of classes to train
    coco_moder : bool
        either the dataset format is coco format or not

    Returns
    -------
    model : nn.Module
        the created model in this case Faster RCNN model
    """

    # Load Faster RCNN pre-trained model
    model = torchvision.models.detection.fasterrcnn_resnet50_fpn_v2(
        weights=torchvision.models.detection.FasterRCNN_ResNet50_FPN_V2_Weights.DEFAULT
    )
    if coco_model:  # Return the COCO pretrained model for COCO classes.
        return model

    # Get the number of input features
    in_features = model.roi_heads.box_predictor.cls_score.in_features
    # define a new head for the detector with required number of classes
    model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)

    return model


def train_faster_rcnn(
    config,
    model: nn.Module,
    train_dataloader: DataLoader,
    optimizer: optim.Optimizer,
) -> nn.Module:
    """Train faster rcnn model

    Parameters
    ----------
    config
        config including hyperparameters
    model: nn.Module
        the created model for the training
    train_dataloader: DataLoader
        dataloader instance for the given dataset which contains images and labels
    optimizer: optim.Optimizer
        optimizer instance for the given model

    Returns
    -------
    model : nn.Module
        the trained model
    """

    # select device (whether GPU or CPU)
    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

    # setup tensorboard stuff
    layout = {
        "Multi": {
            "recon_loss": ["Multiline", ["recon_loss/train", "recon_loss/validation"]],
            "pred_loss": ["Multiline", ["pred_loss/train", "pred_loss/validation"]],
        },
    }
    writer = SummaryWriter(f"/tmp/tensorboard/{int(time.time())}")
    writer.add_custom_scalars(layout)

    def log_scalar(name, value, epoch):
        """Log a scalar value to both MLflow and TensorBoard"""
        writer.add_scalar(name, value, epoch)
        mlflow.log_metric(name, value)

    EXPERIMENT_NAME = "baseline"
    RUN_NAME = f"run_{datetime.today()}"
    print(EXPERIMENT_NAME)

    try:
        EXPERIMENT_ID = mlflow.get_experiment_by_name(EXPERIMENT_NAME).experiment_id
        print(EXPERIMENT_ID)
    except mlflow.exception.MlflowException:
        EXPERIMENT_ID = mlflow.create_experiment(EXPERIMENT_NAME)
        print(EXPERIMENT_ID)

    with mlflow.start_run(experiment_id=EXPERIMENT_ID, run_name=RUN_NAME):
        mlflow.log_params(config)
        mlflow.log_param("n_epoch", config["num_epochs"])

        # # Create a SummaryWriter to write TensorBoard events locally
        output_dir = tempfile.mkdtemp()

        for epoch in range(config["num_epochs"]):
            print(epoch)
            model.train()
            i = 0
            for batch_idx, (imgs, annotations) in enumerate(train_dataloader):
                i += 1
                imgs = list(img.to(device) for img in imgs)
                annotations = [
                    {k: v.to(device) for k, v in t.items()} for t in annotations
                ]

                loss_dict = model(imgs, annotations)
                losses = sum(loss for loss in loss_dict.values())

                optimizer.zero_grad()
                losses.backward()
                optimizer.step()

                print(f"Iteration: {i}/{len(train_dataloader)}, Loss: {losses}")

            log_scalar("total_loss/train", losses, epoch)

        # Upload the TensorBoard event logs as a run artifact
        print("Uploading TensorBoard events as a run artifact...")
        mlflow.log_artifacts(output_dir, artifact_path="events")
        print(
            "\nLaunch TensorBoard with:\n\ntensorboard --logdir=%s"
            % os.path.join(mlflow.get_artifact_uri(), "events")
        )

        # Log the model as an artifact of the MLflow run.
        print("\nLogging the trained model as a run artifact...")
        mlflow.pytorch.log_model(
            model, artifact_path="pytorch-model", pickle_module=pickle
        )
        print(
            "\nThe model is logged at:\n%s"
            % os.path.join(mlflow.get_artifact_uri(), "pytorch-model")
        )
    writer.close()

    return model
