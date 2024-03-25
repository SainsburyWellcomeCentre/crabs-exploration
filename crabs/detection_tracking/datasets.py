import json
import os
import tempfile
from typing import Callable, Optional

import torch
from pycocotools.coco import COCO
from torchvision.datasets import CocoDetection, wrap_dataset_for_transforms_v2


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
        for img_dir, annotation_file in zip(
            list_img_dirs, list_annotation_files
        ):
            # exclude files if required
            if list_exclude_files:
                # check if this annotation files has images to exclude
                coco_obj = COCO(annotation_file)
                n_imgs_to_exclude = sum(
                    [
                        im["file_name"] in list_exclude_files
                        for _, im in coco_obj.imgs.items()
                    ]
                )

                # if it has: create temp file with new annotations
                # and use that to define coco dataset
                if n_imgs_to_exclude > 0:
                    tmp_file, tmp_file_path = tempfile.mkstemp()
                    try:
                        # save modified annotations to tmp file
                        annotation_file = self.create_tmp_annotation_file(
                            annotation_file,
                            list_exclude_files,
                            tmp_file,
                            tmp_file_path,
                        )
                        # create COCO dataset for detection
                        dataset_coco = CocoDetection(
                            img_dir,
                            annotation_file,
                            transforms=transforms,
                        )
                    finally:
                        os.remove(tmp_file_path)
                else:
                    # create COCO dataset for detection
                    dataset_coco = CocoDetection(
                        img_dir,
                        annotation_file,
                        transforms=transforms,
                    )

            else:
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

    def create_tmp_annotation_file(
        self,
        annotation_file: str,
        list_files_to_exclude: list[str],
        tmp_file,
        tmp_file_path,
    ) -> str:
        """Remove selected images from annotation file and save new file.

        A new annotation file is created, excluding the required images,
        and without the annotations for those images.

        Parameters
        ----------
        annotation_file : str
            path to file with annotations
        list_files_to_exclude : list[str]
            list of filenames to exclude from the dataset

        Returns
        -------
        str
            path to new annotation file
        """
        # Read annotation file as a dataset dict
        with open(annotation_file, "r") as f:
            dataset = json.load(f)

        # Determine images to exclude
        image_ids_to_exclude = [
            im["id"]
            for im in dataset["images"]
            if im["file_name"] in list_files_to_exclude
        ]

        assert len(image_ids_to_exclude) > 0

        # Remove required images from dataset
        dataset_imgs = dataset["images"].copy()
        for im in dataset["images"]:
            if im["id"] in image_ids_to_exclude:
                dataset_imgs.remove(im)
        dataset["images"] = dataset_imgs

        # Determine annotations to exclude
        annotation_ids_to_exclude = [
            ann["id"]
            for ann in dataset["annotations"]
            if ann["image_id"] in image_ids_to_exclude
        ]

        # Remove required annotations from dataset
        dataset_annotations = dataset["annotations"].copy()
        for annot in dataset["annotations"]:
            if annot["id"] in annotation_ids_to_exclude:
                dataset_annotations.remove(annot)
        dataset["annotations"] = dataset_annotations

        # Write new annotations to TEMP file
        with os.fdopen(tmp_file, "w") as tmp:
            json.dump(dataset, tmp)

        return str(tmp_file_path)

        # return dataset
