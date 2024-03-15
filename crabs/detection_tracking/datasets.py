import datetime
import json
from pathlib import Path
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

        Example:
        > dataset = CrabsCocoDetection(IMAGES_PATH, ANNOTATIONS_PATH)

        > sample = dataset[0]
        > img, annotations = sample
        > print(type(annotations))  # this is a dictionary

        Should be equivalent to the dataset obtain with:
        > dataset = wrap_dataset_for_transforms_v2(CocoDetection(IMAGES_PATH, ANNOTATIONS_PATH))
        """

        # Create list of transformed-COCO datasets
        list_datasets = []
        for img_dir, annotation_file in zip(
            list_img_dirs, list_annotation_files
        ):
            # exclude files if required
            if list_exclude_files:
                annotation_file = self.exclude_files(
                    annotation_file, list_exclude_files
                )

            # create COCO dataset for detection
            dataset_coco = CocoDetection(
                img_dir,
                annotation_file,
                transforms=transforms,
            )

            # transform for "transforms v2"
            dataset_transformed = wrap_dataset_for_transforms_v2(dataset_coco)
            list_datasets.append(dataset_transformed)

        # Concatenate datasets
        full_dataset = torch.utils.data.ConcatDataset(list_datasets)

        # Update class for this instance
        self.__class__ = full_dataset.__class__
        self.__dict__ = full_dataset.__dict__

    def exclude_files(
        self,
        annotation_file: str,
        list_files_to_exclude: list[str],
    ) -> str:
        """Exclude images from annotation file.

        A new annotation file is created without the images to exclude,
        and without the annotations for that image.

        Parameters
        ----------
        annotation_file : str
            _description_
        list_files_to_exclude : list[str]
            _description_

        Returns
        -------
        str
            _description_
        """
        # read annotation file as a dataset dict
        with open(annotation_file, "r") as f:
            dataset = json.load(f)

        # determine images to exclude
        slc_images_to_exclude = [
            im["file_name"] in list_files_to_exclude
            for im in dataset["images"]
        ]
        image_ids_to_exclude = [
            im["id"]
            for im, slc in zip(dataset["images"], slc_images_to_exclude)
            if slc
        ]

        # determine annotations to exclude
        slc_annotations_to_exclude = [
            ann["image_id"] in image_ids_to_exclude
            for ann in dataset["annotations"]
        ]

        # update dataset dict
        dataset["images"] = [
            im
            for im, slc in zip(dataset["images"], slc_images_to_exclude)
            if not slc
        ]
        dataset["annotations"] = [
            annot
            for annot, slc in zip(
                dataset["annotations"], slc_annotations_to_exclude
            )
            if not slc
        ]

        # write to new file under the same location as original annotation file
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        out_filename = Path(annotation_file).parent / Path(
            Path(annotation_file).stem + "_filt_" + timestamp + ".json"
        )
        with open(out_filename, "w") as f:
            json.dump(dataset, f)

        return str(out_filename)


def plot_sample(imgs, row_title: Optional[str] = None, **imshow_kwargs):
    """
    Plot a sample from the dataset.

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
