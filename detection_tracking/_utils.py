import os

import torch
from PIL import Image
from pycocotools.coco import COCO
import torchvision.transforms as transforms


class myFasterRCNNDataset(torch.utils.data.Dataset):
    """Custom Pytorch dataset class for Faster RCNN object detection using COCO-style annotation.

    Args
    ----------
    data_dir : str
        Path to the directory containing image data.
    annotation : str
        Path to the coco-style annotation file.
    transforms : callable, optional
        A function to apply to the images

    Attributes
    ----------
    data_dir : str
        Path to the directory containing image data.
    annotation : str
        Path to the coco-style annotation file.
    ids : list
        List of image IDs from the COCO annotation.
    transforms : callable, optional
        A function to apply to the images

    Returns
    ----------
    tuple : A tuple containing an image tensor and a dictionary of annotations
    
    """
    def __init__(self, data_dir, annotation, transforms=None):
        self.data_dir = data_dir
        self.coco = COCO(annotation)
        self.transforms = transforms
        self.ids = list(sorted(self.coco.imgs.keys()))

    def __getitem__(self, index):
        """Get the image and associated annotations at the specified index.         

        Args
        ----------
        index : str
            Index of the sample to retrieve.         

        Returns
        ----------        
        tuple: A tuple containing the image tensor and a dictionary of annotations.                     

        Note
        ----------       
        The annotations dictionary contains the following keys:             
            - 'image': The image tensor.             
            - 'annotations': A dictionary containing object annotations with keys:                 
            - 'boxes': Bounding box coordinates (xmin, ymin, xmax, ymax).                 
            - 'labels': Class labels for each object.                 
            - 'image_id': Image ID.                 
            - 'area': Area of each object.                 
        -    'iscrowd': Flag indicating whether the object is a crowd.         
        """

        img_id = self.ids[index]
        # List: get annotation id from coco
        ann_ids = self.coco.getAnnIds(imgIds=img_id)
        # Dictionary: target coco_annotation file for an image
        coco_annotation = self.coco.loadAnns(ann_ids)

        # If there are no annotations, skip this frame (unlabeled frame)
        if not coco_annotation:
            return None

        img_info = self.coco.loadImgs(ids=img_id)[0]
        img_path = os.path.join(self.data_dir, img_info["file_name"])
        # open the input image
        img = Image.open(img_path).convert("RGB")

        # number of objects in the image
        num_objs = len(coco_annotation)

        # Bounding boxes for objects
        # In coco format, bbox = [xmin, ymin, width, height]
        # In pytorch, the input should be [xmin, ymin, xmax, ymax]
        boxes = []
        for i in range(num_objs):
            xmin = coco_annotation[i]["bbox"][0]
            ymin = coco_annotation[i]["bbox"][1]
            xmax = xmin + coco_annotation[i]["bbox"][2]
            ymax = ymin + coco_annotation[i]["bbox"][3]
            boxes.append([xmin, ymin, xmax, ymax])
        boxes = torch.as_tensor(boxes, dtype=torch.float32)
        # Labels (In my case, I only one class: target class or background)
        labels = torch.ones((num_objs,), dtype=torch.int64)
        # Tensorise img_id
        img_id = torch.tensor([img_id])
        # Size of bbox (Rectangulargit stat)
        areas = []
        for i in range(num_objs):
            areas.append(coco_annotation[i]["area"])
        areas = torch.as_tensor(areas, dtype=torch.float32)
        # Iscrowd
        iscrowd = torch.zeros((num_objs,), dtype=torch.int64)

        # Annotation is in dictionary format
        my_annotation = {}
        my_annotation["boxes"] = boxes
        my_annotation["labels"] = labels
        my_annotation["image_id"] = img_id
        my_annotation["area"] = areas
        my_annotation["iscrowd"] = iscrowd
        
        if self.transforms is not None:
            img = self.transforms(img)

        return img, my_annotation

    def __len__(self):
        """Get the total number of samples in the dataset.

        Returns
        ----------
            int: The number of samples in the dataset
        """
        return len(self.ids)

def coco_category():
    COCO_INSTANCE_CATEGORY_NAMES = [
        "__background__",
        "crab",
    ]
    return COCO_INSTANCE_CATEGORY_NAMES

# collate_fn needs for batch
def collate_fn(batch):
    # Filter out None values
    batch = [sample for sample in batch if sample is not None]

    if len(batch) == 0:
        return None

    return tuple(zip(*batch))


def create_dataloader(my_dataset, batch_size):
    # own DataLoader
    data_loader = torch.utils.data.DataLoader(
        my_dataset, batch_size=batch_size, shuffle=True, num_workers=4, collate_fn=collate_fn
    )
    return data_loader


def save_model(model) -> None:
    """Save the model and embeddings"""

    torch.save(model, "pretrain_fasterrcnn_model.pt")
    print("Model Saved")


def get_transform():
    # TODO: testing with different transforms
    custom_transforms = []
    custom_transforms.append(transforms.ToTensor())

    return transforms.Compose(custom_transforms)