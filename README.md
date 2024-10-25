# crabs-exploration

[![License](https://img.shields.io/badge/License-BSD_3--Clause-orange.svg)](https://opensource.org/licenses/BSD-3-Clause)
[![CI](https://img.shields.io/github/actions/workflow/status/SainsburyWellcomeCentre/crabs-exploration/test_and_deploy.yml?label=CI)](https://github.com/SainsburyWellcomeCentre/crabs-exploration/actions/workflows/test_and_deploy.yml)
[![codecov](https://codecov.io/gh/sainsburyWellcomeCentre/crabs-exploration/graph/badge.svg?token=9dM37vnAIT)](https://codecov.io/gh/sainsburyWellcomeCentre/crabs-exploration)

A toolkit for detecting and tracking crabs in the field.

## Getting Started

### Prerequisites

<!-- Any tools or versions of languages needed to run code. For example specific Python or Node versions. Minimum hardware requirements also go here. -->

`crabs` uses neural networks to detect and track multiple crabs in the field. The detection model is based on the [Faster R-CNN](https://arxiv.org/abs/1506.01497) architecture. The tracking model is based on the [SORT](https://github.com/abewley/sort) tracking algorithm.

The package supports Python 3.9 or 3.10, on Linux and MacOS.

Running `crabs` on a machine with a dedicated graphics device is highly recommended. This usually means an NVIDIA GPU or Apple M1 chip.


### Installation

#### Users
To install the `crabs` package, first clone this git repository.
```bash
git clone https://github.com/SainsburyWellcomeCentre/crabs-exploration.git
```

Then, navigate to the root directory of the repository and install the `crabs` package in a conda environment:

```bash
conda create -n crabs-env python=3.10 -y
conda activate crabs-env
pip install .
```

#### Developers
For development, we recommend installing the package in editable mode and with additional `dev` dependencies:

```bash
pip install -e .[dev]  # or ".[dev]" if you are using zsh
```

### CrabsField - Sept2023 dataset

We trained the detector model on our [CrabsField - Sept2023](https://gin.g-node.org/SainsburyWellcomeCentre/CrabsField) dataset. The dataset consists of 53041 annotations (bounding boxes) over 544 frames extracted from 28 videos of crabs in the field.

The dataset is currently private. If you have access to the [GIN](https://gin.g-node.org/) repository, you can download the dataset using the GIN CLI tool. To set up the GIN CLI tool:
1. Create [a GIN account](https://gin.g-node.org/user/sign_up).
2. [Download GIN CLI](https://gin.g-node.org/G-Node/Info/wiki/GIN+CLI+Setup#setup-gin-client) and set it up by running:
   ```
   $ gin login
   ```
   You will be prompted for your GIN username and password.
3. Confirm that everything is working properly by typing:
   ```
   $ gin --version
   ```

Then to download the dataset, run:
```
gin get SainsburyWellcomeCentre/CrabsField
```
This command will clone the data repository to the current working directory, and download the large files in the dataset as lightweight placeholders. To download the content of these placeholder files, run:
```
gin download --content
```
Because the large files in the dataset are **locked**, this command will download the content to the git annex subdirectory, and turn the placeholder files in the working directory into symlinks that point to the content. For more information on how to work with a GIN repository, see the [NIU HowTo guide](https://howto.neuroinformatics.dev/open_science/GIN-repositories.html).

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

We use [MLflow](https://mlflow.org) to monitor the training of the detector and log the hyperparameters used.

To run MLflow, execute the following command from your `crabs-env` conda environment:

```
mlflow ui --backend-store-uri file:///<path-to-ml-runs>
```

Replace `<path-to-ml-runs>` with the path to the directory where the MLflow output is. By default, the output is in an `ml-runs` directory under the current working directory.

### Evaluate a detector

To evaluate a trained detector on the test split of the dataset, run the following command:

```
evaluate-detector --model_dir {directory_to_trained_model} --dataset_dirs {parent_directory_of_frames_and_annotation}
```

This command assumes the dataset structure described on the previous section. It will output the main evaluation metrics for the detector.

### Run detector+tracking on a video

To track crabs in a new video, using a trained detector + tracker, run the following command:

```
track-video --model_dir {path_to_trained_model} --vid_path {path_to_input_video}
```

This will produce a .csv file with the tracking results, that can be imported in [movement](https://github.com/neuroinformatics-unit/movement) for further analysis. See the [movement documentation](https://movement.neuroinformatics.dev/getting_started/input_output.html#loading-bounding-boxes-tracks) for more details.

<!-- ### Evaluate the tracking performance

To evaluate the tracking performance of a trained detector + tracker, run the following command:

```
evaluate-tracking ...
```

We currently only support the SORT tracker, and the evaluation is based on the MOTA metric. -->

<!-- # Other common workflows -->
<!-- [TODO: add separate guides for this? eventually make into sphinx docs?] -->
<!-- - Prepare data for training a detector -->
  <!-- - Extract frames from videos -->
  <!-- - Annotate the frames with bounding boxes -->
  <!-- - Combine several annotation files into a single file -->
<!-- - Retrain a detector on an extended dataset -->
<!-- - Prepare data for labelling ground truth for tracking -->
