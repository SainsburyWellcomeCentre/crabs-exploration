import datetime
import json
from pathlib import Path
from typing import Callable, Optional

import torch
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
        select_images_to_exclude = [
            im["file_name"] in list_files_to_exclude
            for im in dataset["images"]
        ]

        # If there are no images to exclude: return
        # the original annotation file
        if not any(select_images_to_exclude):
            return annotation_file
        # else create a new one
        else:
            image_ids_to_exclude = [
                im["id"]
                for im, sel in zip(dataset["images"], select_images_to_exclude)
                if sel
            ]

            # Determine annotations to exclude
            select_annotations_to_exclude = [
                ann["image_id"] in image_ids_to_exclude
                for ann in dataset["annotations"]
            ]

            # Update dataset dict
            dataset["images"] = [
                im
                for im, sel in zip(dataset["images"], select_images_to_exclude)
                if not sel
            ]
            dataset["annotations"] = [
                annot
                for annot, sel in zip(
                    dataset["annotations"], select_annotations_to_exclude
                )
                if not sel
            ]

            # Write to new file under the same location as original annotation file
            timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
            out_filename = Path(annotation_file).parent / Path(
                Path(annotation_file).stem + "_filt_" + timestamp + ".json"
            )
            with open(out_filename, "w") as f:
                json.dump(dataset, f)

            return str(out_filename)
