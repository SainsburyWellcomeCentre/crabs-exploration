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
    file_path = main_dir_path / "images" / file_name
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

    Attributes
    ----------
    main_dir : str
        Path to the main directory containing the data.
    train_file_path : list
        A List containing str for path of the training files.
    coco : Object
        Instance of COCO object for handling annotations.
    ids : list
        List of image IDs from the COCO annotation.
    transforms : callable, optional
        A function to apply to the images

    Returns
    ----------
    tuple
        A tuple containing an image tensor and a dictionary of annotations

    """

    def __init__(
        self, main_dir, train_file_paths, annotation, transforms=None
    ):
        self.main_dir = main_dir
        self.file_paths = train_file_paths
        self.coco = COCO(annotation)
        self.transforms = transforms
        self.ids = list(sorted(self.coco.imgs.keys()))

    def __getitem__(self, index):
        """Get the image and associated annotations at the specified index.

        Parameters
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
            - 'iscrowd': Flag indicating whether the object is a crowd.
        - In coco format, bbox = [xmin, ymin, width, height]
        - In pytorch, the input should be [xmin, ymin, xmax, ymax]
        """

        file_name = self.file_paths[index]
        file_path = get_file_path(self.main_dir, file_name)
        img = Image.open(file_path).convert("RGB")

        img_id = [
            img_info["id"]
            for img_info in self.coco.imgs.values()
            if img_info["file_name"] == file_name
        ][0]

        ann_ids = self.coco.getAnnIds(imgIds=img_id)
        coco_annotation = self.coco.loadAnns(ann_ids)

        if not coco_annotation:
            return None

        num_objs = len(coco_annotation)

        boxes = []
        for i in range(num_objs):
            xmin = coco_annotation[i]["bbox"][0]
            ymin = coco_annotation[i]["bbox"][1]
            xmax = xmin + coco_annotation[i]["bbox"][2]
            ymax = ymin + coco_annotation[i]["bbox"][3]
            boxes.append([xmin, ymin, xmax, ymax])
        boxes = torch.as_tensor(boxes, dtype=torch.float32)
        labels = torch.ones((num_objs,), dtype=torch.int64)
        img_id = torch.tensor([img_id])

        areas = []
        for i in range(num_objs):
            areas.append(coco_annotation[i]["area"])

        areas = torch.as_tensor(areas, dtype=torch.float32)
        iscrowd = torch.zeros((num_objs,), dtype=torch.int64)

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
        return len(self.file_paths)
