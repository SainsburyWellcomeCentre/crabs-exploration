import cv2
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
from _utils import coco_category
from torch.utils.data import DataLoader
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
    num_epochs: int,
    model: nn.Module,
    train_dataloader: DataLoader,
    optimizer: optim.Optimizer,
) -> nn.Module:
    """Train faster rcnn model

    Parameters
    ----------
    num_epochs : int
        number of total epochs for the training
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

    model.train()
    for epoch in range(num_epochs):
        print(epoch)
        i = 0
        for imgs, annotations in train_dataloader:
            i += 1
            imgs = list(img.to(device) for img in imgs)
            annotations = [{k: v.to(device) for k, v in t.items()} for t in annotations]

            loss_dict = model(imgs, annotations)
            losses = sum(loss for loss in loss_dict.values())

            optimizer.zero_grad()
            losses.backward()
            optimizer.step()

            print(f"Iteration: {i}/{len(train_dataloader)}, Loss: {losses}")

    return model
