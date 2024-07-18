# crabs-exploration

[![License](https://img.shields.io/badge/License-BSD_3--Clause-orange.svg)](https://opensource.org/licenses/BSD-3-Clause)
[![CI](https://img.shields.io/github/actions/workflow/status/SainsburyWellcomeCentre/crabs-exploration/test_and_deploy.yml?label=CI)](https://github.com/SainsburyWellcomeCentre/crabs-exploration/actions/workflows/test_and_deploy.yml)
[![codecov](https://codecov.io/gh/sainsburyWellcomeCentre/crabs-exploration/graph/badge.svg?token=9dM37vnAIT)](https://codecov.io/gh/sainsburyWellcomeCentre/crabs-exploration)

A toolkit for detecting and tracking crabs in the field.

## Getting Started

### Prerequisites

<!-- Any tools or versions of languages needed to run code. For example specific Python or Node versions. Minimum hardware requirements also go here. -->

requires Python 3.9 or 3.10 or 3.11.

### Installation

<!-- How to build or install the application. -->

### Data Structure

We assume the following structure for the dataset directory:

```
|_ Dataset
    |_ frames
    |_ annotations
        |_ VIA_JSON_combined_coco_gen.json
```

The default name assumed for the annotations file is `VIA_JSON_combined_coco_gen.json`. This is used if no input files are passed. Other filenames (or fullpaths) can be passed with the `--annotation_files` command-line argument.

### Running Locally

For training

```bash
python train-detector --dataset_dirs {parent_directory_of_frames_and_annotation} {optional_second_parent_directory_of_frames_and_annotation} --annotation_files {path_to_annotation_file.json} {path_to_optional_second_annotation_file.json}
```

Example (using default annotation file and one dataset):

```bash
python train-detector --dataset_dirs /home/data/dataset1
```

Example (passing the full path of the annotation file):

```bash
python train-detector --dataset_dirs /home/data/dataset1 --annotation_files /home/user/annotations/annotations42.json
```

Example (passing several datasets with annotation filenames different from the default):

```bash
python train-detector --dataset_dirs /home/data/dataset1 /home/data/dataset2 --annotation_files annotation_dataset1.json annotation_dataset2.json
```

For evaluation

```bash
python evaluate-detector --model_dir {directory_to_saved_model} --images_dirs {parent_directory_of_frames_and_annotation} {optional_second_parent_directory_of_frames_and_annotation} --annotation_files {annotation_file.json} {optional_second_annotation_file.json}
```

Example:

```bash
python evaluate-detector --model_dir model/model_00.pt --main_dir /home/data/dataset1/frames /home/data/dataset2/frames --annotation_files /home/data/dataset1/annotations/annotation_dataset1.json /home/data/dataset2/annotations/annotation_dataset2.json
```

For running inference

```bash
python crabs/detection_tracking/inference_model.py --model_dir {oath_to_trained_model} --vid_path {path_to_input_video}
```

### MLFLow

We are using [MLflow](https://mlflow.org) to log our training loss and the hyperparameters used.
To run MLflow, execute the following command in your terminal:

```
mlflow ui --backend-store-uri file:///<path-to-ml-runs>
```

Replace `<path-to-ml-runs>` with the path to the directory where you want to store the MLflow output. By default, it's an `ml-runs` directory under the current working directory.
