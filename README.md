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
```

### Running Locally

For training

```bash
python crabs/detection_tracking/train_model.py --images_dirs {parent_directory_of_frames_and_annotation} {optional_second_parent_directory_of_frames_and_annotation} --annotation_files {path_to_annotation_file.json} {path_to_optional_second_annotation_file.json}
```

Example:

```bash
python crabs/detection_tracking/evaluate_model.py --main_dir /home/data/dataset1/frames /home/data/dataset2/frames --annotation_file /home/data/dataset2/annotations/annotation_dataset1.json /home/data/dataset2/frames/annotations/annotation_dataset2.json
```

For evaluation

```bash
python crabs/detection_tracking/evaluate_model.py --model_dir {directory_to_saved_model} --main_dir {parent_directory_of_frames_and_annotation} {optional_second_parent_directory_of_frames_and_annotation} --annotation_file {annotation_file.json} {optional_second_annotation_file.json}
```

Example:

```bash
python crabs/detection_tracking/evaluate_model.py --model_dir model/model_00.pt --main_dir /home/data/dataset1 /home/data/dataset2 --annotation_file annotation_dataset1.json annotation_dataset2.json
```

For running inference

```bash
python crabs/detection_tracking/inference_model.py --model_dir {oath_to_trained_model} --vid_path {path_to_input_video}
```
