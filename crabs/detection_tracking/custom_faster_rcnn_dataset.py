import os
from pathlib import Path

import torch
from PIL import Image
from pycocotools.coco import COCO


def get_file_path(main_dir, file_name) -> str:
    """
    Get the file path by joining the main directory and file name.

    Parameters
    ----------
    main_dir : str
        Main directory path.
    file_name :str
        Name of the file.

    Returns
    ----------
    file path : str
        File path joining the main directory and file_name of images to create the full image path.
    """
    main_dir_path = Path(__file__).parent.parent.joinpath(main_dir)
    file_path = main_dir_path / "frames" / file_name
    return str(file_path)


class CustomFasterRCNNDataset(torch.utils.data.Dataset):
    """Custom Pytorch dataset class for Faster RCNN object detection
    using COCO-style annotation.

    Parameters
    ----------
    main_dir : str
        Path to the main directory containing the data.
    train_file_path : list
        A List containing str for path of the training files.
    annotation : str
        Path to the coco-style annotation file.
    transforms : callable, optional
        A function to apply to the images

    Returns
    ----------
    tuple
        A tuple containing an image tensor and a dictionary of annotations

    """

    def __init__(self, file_paths, annotations, transforms=None):
        self.file_paths = file_paths
        self.annotations = [COCO(annotation) for annotation in annotations]
        self.transforms = transforms

    def __getitem__(self, index):
        """Get the image and associated annotations at the specified index.

        Parameters
        ----------
        index : str
            Index of the sample to retrieve.

        Returns
        -------
        tuple: A tuple containing the image tensor and a dictionary of annotations.

        Notes
        -----
        The annotations dictionary contains the following keys:
            - 'image': The image tensor.
            - 'annotations': A dictionary containing object annotations with keys:
            - 'boxes': Bounding box coordinates (xmin, ymin, xmax, ymax).
            - 'labels': Class labels for each object.
            - 'image_id': Image ID.
            - 'area': Area of each object.
            - 'iscrowd': Flag indicating whether the object is a crowd.
        - In coco format, bbox = [xmin, ymin, width, height]
        - In pytorch, the input should be [xmin, ymin, xmax, ymax]
        """

        file_name = os.path.basename(self.file_paths[index])
        img = Image.open(self.file_paths[index]).convert("RGB")

        coco_annotations = []
        for annotation in self.annotations:
            img_id = [
                img_info["id"]
                for img_info in annotation.imgs.values()
                if img_info["file_name"] == file_name
            ]
            if not img_id:
                coco_annotations.append(None)
            else:
                ann_ids = annotation.getAnnIds(imgIds=img_id[0])
                coco_annotations.append(annotation.loadAnns(ann_ids))

        combined_annotations = []
        for annotations in coco_annotations:
            if annotations:
                for ann in annotations:
                    xmin = ann["bbox"][0]
                    ymin = ann["bbox"][1]
                    xmax = xmin + ann["bbox"][2]
                    ymax = ymin + ann["bbox"][3]
                    boxes = [xmin, ymin, xmax, ymax]
                    combined_annotations.append(
                        {
                            "boxes": boxes,
                            "labels": ann["category_id"],
                            "image_id": img_id,
                            "area": ann["area"],
                            "iscrowd": ann["iscrowd"],
                        }
                    )

        # Convert to tensors
        boxes = torch.tensor(
            [ann["boxes"] for ann in combined_annotations], dtype=torch.float32
        )
        labels = torch.tensor(
            [ann["labels"] for ann in combined_annotations], dtype=torch.int64
        )
        img_id = torch.tensor(
            [ann["image_id"] for ann in combined_annotations],
            dtype=torch.int64,
        )
        areas = torch.tensor(
            [ann["area"] for ann in combined_annotations], dtype=torch.float32
        )
        iscrowd = torch.tensor(
            [ann["iscrowd"] for ann in combined_annotations], dtype=torch.int64
        )

        my_annotation = {
            "boxes": boxes,
            "labels": labels,
            "image_id": img_id,
            "area": areas,
            "iscrowd": iscrowd,
        }

        if self.transforms is not None:
            img = self.transforms(img)

        return img, my_annotation

    def __len__(self):
        """Get the total number of samples in the dataset.

        Returns
        ----------
            int: The number of samples in the dataset
        """
        return len(self.file_paths)
