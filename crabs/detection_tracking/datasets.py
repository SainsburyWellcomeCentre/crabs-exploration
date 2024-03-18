from typing import Callable, Optional

import matplotlib.pyplot as plt
import torch
from torchvision import tv_tensors
from torchvision.datasets import CocoDetection, wrap_dataset_for_transforms_v2
from torchvision.transforms.v2 import functional as F
from torchvision.utils import draw_bounding_boxes, draw_segmentation_masks


class CrabsCocoDetection(torch.utils.data.ConcatDataset):
    def __init__(
        self,
        list_img_dirs: list[str],
        list_annotation_files: list[str],
        transforms: Optional[Callable] = None,
        list_exclude_files: Optional[list[str]] = None,
    ):
        """
        CocoDetection dataset wrapped for transforms_v2.

        Example usage:
        > dataset = CrabsCocoDetection([IMAGES_PATH], [ANNOTATIONS_PATH])
        > sample = dataset[0]
        > img, annotations = sample
        > print(type(annotations))  # this is a dictionary

        Should produce a dataset equivalent to one obtained with:
        > dataset = wrap_dataset_for_transforms_v2(CocoDetection([IMAGES_PATH], [ANNOTATIONS_PATH]))
        """

        # Create list of transformed-COCO datasets
        list_datasets = []
        dict_cocos = {}
        for img_dir, annotation_file in zip(
            list_img_dirs, list_annotation_files
        ):
            # create COCO dataset for detection
            dataset_coco = CocoDetection(
                img_dir,
                annotation_file,
                transforms=transforms,
            )

            # exclude ids of certain files if required
            if list_exclude_files:
                dataset_coco = self.exclude_files(
                    dataset_coco, list_exclude_files
                )

            # transform for "transforms v2"
            dataset_transformed = wrap_dataset_for_transforms_v2(dataset_coco)
            list_datasets.append(dataset_transformed)

            # add COCO object of this dataset to dictionary
            dict_cocos[str(img_dir)] = dataset_transformed.coco

        # Concatenate datasets
        full_dataset = torch.utils.data.ConcatDataset(list_datasets)

        # Update class for this instance
        self.__class__ = full_dataset.__class__
        self.__dict__ = full_dataset.__dict__

        # add dictionary of COCO objects per dataset as an attribute
        # this doesnt account for any excluded files!
        self._coco_obj_per_dataset = dict_cocos

    def exclude_files(
        self,
        dataset_coco,
        list_files_to_exclude: list[str],
    ) -> str:
        """Remove selected images from annotation file and save new file.

        A new annotation file is created, excluding the required images,
        and without the annotations for those images.

        Parameters
        ----------
        annotation_file : str
            file with annotations
        list_files_to_exclude : list[str]
            list of filenames to exclude from the dataset

        Returns
        -------
        str
            path to new annotation file
        """

        # map filename to image ID
        map_filename_to_img_id = {
            val["file_name"]: val["id"]
            for ky, val in dataset_coco.coco.imgs.items()
        }

        # find image IDs to keep/remove files
        img_ids_to_remove = [
            map_filename_to_img_id[k] for k in list_files_to_exclude
        ]
        img_ids_to_keep = [
            id for id in dataset_coco.ids if id not in img_ids_to_remove
        ]

        # overwrite indices
        # self.dataset = dataset
        # self.createIndex()
        dataset_coco.ids = img_ids_to_keep

        return dataset_coco


def plot_sample(imgs: list, row_title: Optional[str] = None, **imshow_kwargs):
    """
    Plot a sample (image & annotations) from a dataset.

    Example usage:
    > full_dataset = CrabsCocoDetection([IMAGES_PATH],[ANNOTATIONS_PATH])
    > sample = full_dataset[0]
    > plt.figure()
    > plot_sample([sample])

    From https://github.com/pytorch/vision/blob/main/gallery/transforms/helpers.py
    """
    if not isinstance(imgs[0], list):
        # Make a 2d grid even if there's just 1 row
        imgs = [imgs]

    num_rows = len(imgs)
    num_cols = len(imgs[0])
    _, axs = plt.subplots(nrows=num_rows, ncols=num_cols, squeeze=False)
    for row_idx, row in enumerate(imgs):
        for col_idx, img in enumerate(row):
            boxes = None
            masks = None
            if isinstance(img, tuple):
                img, target = img
                if isinstance(target, dict):
                    boxes = target.get("boxes")
                    masks = target.get("masks")
                elif isinstance(target, tv_tensors.BoundingBoxes):
                    boxes = target
                else:
                    raise ValueError(f"Unexpected target type: {type(target)}")
            img = F.to_image(img)
            if img.dtype.is_floating_point and img.min() < 0:
                # Poor man's re-normalization for the colors to be OK-ish. This
                # is useful for images coming out of Normalize()
                img -= img.min()
                img /= img.max()

            img = F.to_dtype(img, torch.uint8, scale=True)
            if boxes is not None:
                img = draw_bounding_boxes(img, boxes, colors="yellow", width=3)
            if masks is not None:
                img = draw_segmentation_masks(
                    img,
                    masks.to(torch.bool),
                    colors=["green"] * masks.shape[0],
                    alpha=0.65,
                )

            ax = axs[row_idx, col_idx]
            ax.imshow(img.permute(1, 2, 0).numpy(), **imshow_kwargs)
            ax.set(xticklabels=[], yticklabels=[], xticks=[], yticks=[])

    if row_title is not None:
        for row_idx in range(num_rows):
            axs[row_idx, 0].set(ylabel=row_title[row_idx])

    plt.tight_layout()
