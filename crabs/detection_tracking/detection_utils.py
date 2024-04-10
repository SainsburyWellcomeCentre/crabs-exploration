from pathlib import Path

DEFAULT_ANNOTATIONS_FILENAME = "VIA_JSON_combined_coco_gen.json"


def prep_img_directories(dataset_dirs: list[str]) -> list[str]:
    """
    Derive list of input image directories from a list of dataset directories.
    We assume a specific structure for the dataset directories.

    Parameters:
    -----------
    dataset_dirs : List[str]
        List of directories containing dataset folders.

    Returns:
    --------
    List[str]:
        List of directories containing image frames.
    """
    images_dirs = []
    for dataset in dataset_dirs:
        images_dirs.append(str(Path(dataset) / "frames"))
    return images_dirs


def prep_annotation_files(
    input_annotation_files: list[str], dataset_dirs: list[str]
) -> list[str]:
    """
    Prepares annotation files for processing.

    Parameters:
    -----------
    input_annotation_files : List[str]
        List of annotation files or filenames.
    dataset_dirs : List[str]
        List of directories containing dataset folders.

    Returns:
    --------
    List[str]:
        List of annotation file paths.
    """
    # prepare list of annotation files
    annotation_files = []

    # if none are passed: assume default filename for annotations,
    # and default location under `annotations` directory
    if not input_annotation_files:
        for dataset in dataset_dirs:
            annotation_files.append(
                str(
                    Path(dataset)
                    / "annotations"
                    / DEFAULT_ANNOTATIONS_FILENAME
                )
            )

    # if a list of annotation files/filepaths is passed
    else:
        for annot, dataset in zip(input_annotation_files, dataset_dirs):
            # if the annotation is only filename:
            # assume file is under 'annotation' directory
            if Path(annot).name == annot:
                annotation_files.append(
                    str(Path(dataset) / "annotations" / annot)
                )
            # otherwise assume the full path to the annotations file is passed
            else:
                annotation_files.append(annot)

    return annotation_files
