import torch
import torchvision
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
import torch.nn as nn
from torch.utils.data import DataLoader
import torch.optim as optim
from _utils import coco_category


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
        num_epochs: int, model: nn.Module, train_dataloader: DataLoader, optimizer: optim.Optimizer
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
    for epoch in range(num_epochs):
        model.train()
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


def valid_model(self, trained_model) -> None:

        device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
        total_correct_boxes = 0
        total_gt_boxes = 0

        coco_list = coco_category()

        with torch.no_grad():
            imgs_id = 0
            for imgs, annotations in self.valid_dataloader:
                imgs_id += 1
                imgs = list(img.to(device) for img in imgs)
                targets = [{k: v.to(device) for k, v in t.items()} for t in annotations]
                
                detections = trained_model(imgs, annotations)

                for image, label, prediction in zip(imgs, annotations, detections):
                    image = image.cpu().numpy().transpose(1, 2, 0)
                    image = (image * 255).astype('uint8')
                    image_with_boxes = image.copy()

                    pred_score = list(prediction["scores"].detach().cpu().numpy())

                    target_boxes = [[(i[0], i[1]), (i[2], i[3])] for i in list(label["boxes"].detach().cpu().detach().numpy())]
                    if pred_score:
                        pred_t = [pred_score.index(x) for x in pred_score][-1]

                        pred_class = [coco_list[i] for i in list(prediction["labels"].detach().cpu().numpy())]
                        pred_boxes = [
                            [(i[0], i[1]), (i[2], i[3])]
                            for i in list(prediction["boxes"].detach().cpu().detach().numpy())
                        ]

                        pred_boxes = pred_boxes[: pred_t + 1]
                        pred_class = pred_class[: pred_t + 1]

                        for i in range(len(pred_boxes)):
                            if (pred_class[i]) == "crab" and pred_score[i] > self.config["score_threshold"]:
                                
                                cv2.rectangle(
                                    image_with_boxes,
                                    (int((pred_boxes[i][0])[0]), int((pred_boxes[i][0])[1])),
                                    (int((pred_boxes[i][1])[0]), int((pred_boxes[i][1])[1])),
                                    (0, 0, 255),
                                    2,
                                )

                                label_text = f"{pred_class[i]}: {pred_score[i]:.2f}"
                                # id_label = f"id : {sort_bbs_ids[i][4]}"
                                cv2.putText(
                                    image_with_boxes,
                                    label_text,
                                    (int((pred_boxes[i][0])[0]), int((pred_boxes[i][0])[1])),
                                    cv2.FONT_HERSHEY_SIMPLEX,
                                    1,
                                    (0, 0, 255),
                                    thickness=2,
                                )

                        for i in range(len(target_boxes)):
                            
                            cv2.rectangle(
                                image_with_boxes,
                                (int((target_boxes[i][0])[0]), int((target_boxes[i][0])[1])),
                                (int((target_boxes[i][1])[0]), int((target_boxes[i][1])[1])),
                                (0, 255, 0),
                                2,
                            )
                            
                        cv2.imwrite(f"imgs{imgs_id}.jpg", image_with_boxes)

                for target, detection in zip(targets, detections):
                    gt_boxes = target["boxes"]
                    pred_boxes = detection["boxes"]

                    # compare predicted boxes to ground truth boxes
                    ious = torchvision.ops.box_iou(pred_boxes, gt_boxes)
                    correct_boxes = (ious > self.config["score_threshold"]).sum().item()
                    total_correct_boxes += correct_boxes
                    total_gt_boxes += len(gt_boxes)

        average_precision = total_correct_boxes / total_gt_boxes
        print(f"Average Precision: {average_precision:.4f}")