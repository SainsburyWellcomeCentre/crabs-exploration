# crabs-exploration

[![License](https://img.shields.io/badge/License-BSD_3--Clause-orange.svg)](https://opensource.org/licenses/BSD-3-Clause)
[![CI](https://img.shields.io/github/actions/workflow/status/SainsburyWellcomeCentre/crabs-exploration/test_and_deploy.yml?label=CI)](https://github.com/SainsburyWellcomeCentre/crabs-exploration/actions/workflows/test_and_deploy.yml)
[![codecov](https://codecov.io/gh/sainsburyWellcomeCentre/crabs-exploration/graph/badge.svg?token=9dM37vnAIT)](https://codecov.io/gh/sainsburyWellcomeCentre/crabs-exploration)

A toolkit for detecting and tracking crabs in the field.

## Getting Started

### Prerequisites

We support Python 3.9 or 3.10. The package is tested on Linux and MacOS.

Recommended hardware requirements:
[TODO: minimum GPU specs?]

### Installation

To install the `crabs` package, first create a Python virtual environment. You can use `conda` for this:

```
conda create -n crabs-env python=3.10
conda activate crabs-env
```

Next, install the `crabs` package in the environment. Clone the `crabs-exploration` repository to get the source code locally, and use `pip` to install the package:

```
git clone https://github.com/SainsburyWellcomeCentre/crabs-exploration.git
cd crabs-exploration
pip install .
```

## Basic commands

### Train a detector

To train a detector on an existing dataset, run the following command:

```
train-detector --dataset_dirs {parent_directory_of_frames_and_annotation}
```

This command assumes the following structure for the dataset directory:

```
dataset
|_ frames
|_ annotations
    |_ VIA_JSON_combined_coco_gen.json
```

The default name assumed for the annotations file is `VIA_JSON_combined_coco_gen.json`. Other filenames (or full paths to files) can be passed with the `--annotation_files` command-line argument. See the [Training Models Locally](crabs/guides/TrainingModelsLocally.md) for more details.

### Monitor a training job

We use MLflow [MLflow](https://mlflow.org) to monitor the training of the detector and log the hyperparameters used.

To run MLflow, execute the following command from your `crabs` environment:

```
mlflow ui --backend-store-uri file:///<path-to-ml-runs>
```

Replace `<path-to-ml-runs>` with the path to the directory where the MLflow output is. By default, the output is in an `ml-runs` directory under the current working directory. [TODO: check this]

### Evaluate a detector

To evaluate a trained detector on the test fraction of the dataset, run the following command:

```
evaluate-detector --model_dir {directory_to_trained_model} --dataset_dirs {parent_directory_of_frames_and_annotation}
```

This command assumes the dataset structure described on the previous section. It will produce evaluation metrics for the detector. [TODO: which metrics? how are they exported?]

### Run detector+tracking on a video

To track crabs in a new video, using a trained detector + tracker, run the following command:

```
track-video --model_dir {path_to_trained_model} --vid_path {path_to_input_video}
```

This will produce a .csv file with the tracking results, that can be imported in [movement](https://github.com/neuroinformatics-unit/movement) for further analysis.

### Evaluate the tracking performance

To evaluate the tracking performance of a trained detector + tracker, run the following command:

```
evaluate-tracking ...
```

We currently only support the SORT tracker, and the evaluation is based on the MOTA metric.

# Other common workflows

[TODO: add separate guides for this? eventually make into sphinx docs?]

- Prepare data for training a detector
  - Extract frames from videos
  - Annotate the frames with bounding boxes
  - Combine the annotations into a single file
- Retrain a detector on an extended dataset
- Prepare data for labelling tracking ground truth
