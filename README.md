# crabs-exploration

[![License](https://img.shields.io/badge/License-BSD_3--Clause-orange.svg)](https://opensource.org/licenses/BSD-3-Clause)
[![CI](https://img.shields.io/github/actions/workflow/status/SainsburyWellcomeCentre/crabs-exploration/test_and_deploy.yml?label=CI)](https://github.com/SainsburyWellcomeCentre/crabs-exploration/actions/workflows/test_and_deploy.yml)
[![codecov](https://codecov.io/gh/sainsburyWellcomeCentre/crabs-exploration/graph/badge.svg?token=9dM37vnAIT)](https://codecov.io/gh/sainsburyWellcomeCentre/crabs-exploration)

A toolkit for detecting and tracking crabs in the field.

## Getting Started

### Prerequisites

<!-- Any tools or versions of languages needed to run code. For example specific Python or Node versions. Minimum hardware requirements also go here. -->

`crabs` uses neural networks to detect and track multiple crabs in the field. The detection model is based on the [Faster R-CNN](https://arxiv.org/abs/1506.01497) architecture. The tracking model is based on the [SORT](https://github.com/abewley/sort) tracking algorithm.

The package supports Python 3.9 or 3.10, and is tested on Linux and MacOS.

We highly recommend running `crabs` on a machine with a dedicated graphics device, such as an NVIDIA GPU or an Apple M1+ chip.


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

Then to download the dataset, run the following command from the directory you want the data to be in:
```
gin get SainsburyWellcomeCentre/CrabsField
```
This command will clone the data repository to the current working directory, and download the large files in the dataset as lightweight placeholder files. To download the content of these placeholder files, run:
```
gin download --content
```
Because the large files in the dataset are **locked**, this command will download the content to the git annex subdirectory, and turn the placeholder files in the working directory into symlinks that point to that content. For more information on how to work with a GIN repository, see the corresponding [NIU HowTo guide](https://howto.neuroinformatics.dev/open_science/GIN-repositories.html).

## Basic commands

### Train a detector

To train a detector on an existing dataset, run the following command:

```
train-detector --dataset_dirs <list-of-dataset-directories>
```

This command assumes each dataset directory has the following structure:

```
dataset
|_ frames
|_ annotations
    |_ VIA_JSON_combined_coco_gen.json
```

The default name assumed for the annotations file is `VIA_JSON_combined_coco_gen.json`. Other filenames (or full paths to annotation files) can be passed with the `--annotation_files` command-line argument.

To see the full list of possible arguments to the `train-detector` command run:
```
train-detector --help
```

### Monitor a training job

We use [MLflow](https://mlflow.org) to monitor the training of the detector and log the hyperparameters used.

To run MLflow, execute the following command from your `crabs-env` conda environment:

```
mlflow ui --backend-store-uri file:///<path-to-ml-runs>
```

Replace `<path-to-ml-runs>` with the path to the directory where the MLflow output is. By default, the output is placed in an `ml-runs` folder under the directory from which the `train-detector` is launched.

In the MLflow browser-based user-interface, you can find the path to the checkpoints directory for any run, under the `path_to_checkpoints` parameter. This will be useful to evaluate the trained model. The model saved at the end of the training job is saved as `last.ckpt` in the `path_to_checkpoints` directory.

### Evaluate a detector

To evaluate a trained detector on the test split of the dataset, run the following command:

```
evaluate-detector --trained_model_path <path-to-ckpt-file>
```

This command assumes the trained detector model (a `.ckpt` checkpoint file) is saved in an MLflow database structure. That is, the checkpoint is assumed to be under a `checkpoints` directory, which in turn should be under a `<mlflow-experiment-hash>/<mlflow-run-hash>` directory. This will be the case if the model has been trained using the `train-detector` command.

The `evaluate-detector` command will print to screen the average precision and average recall of the detector on the validation set by default. To evaluate the model on the test set instead, use the `--use_test_set` flag.

The command will also log those performance metrics to the MLflow database, along with the hyperparameters of the evaluation job. To visualise the MLflow summary of the evaluation job, run:
```
mlflow ui --backend-store-uri file:///<path-to-ml-runs>
```
where `<path-to-ml-runs>` is the path to the directory where the MLflow output is.

The evaluated samples can be inspected visually by exporting them using the `--save__frames` flag. In this case, the frames with the predicted and ground-truth bounding boxes are saved in a directory called `evaluation_output_<timestamp>` under the current working directory.

To see the full list of possible arguments to the `evaluate-detector` command, run it with the `--help` flag.

### Run detector+tracking on a video

To track crabs in a new video, using a trained detector and a tracker, run the following command:

```
detect-and-track-video --trained_model_path <path-to-ckpt-file> --video_path <path-to-input-video>
```

This will produce a `tracking_output_<timestamp>` directory with the output from tracking under the current working directory.

The tracking output consists of:
- a .csv file named `<video-name>_tracks.csv`, with the tracked bounding boxes data;
- if the flag `--save_video` is added to the command: a video file named `<video-name>_tracks.mp4`, with the tracked bounding boxes;
- if the flag `--save_frames` is added to the command: a subdirectory named `<video_name>_frames` is created, and the video frames are saved in it.

The .csv file with tracked bounding boxes can be imported in [movement](https://github.com/neuroinformatics-unit/movement) for further analysis. See the [movement documentation](https://movement.neuroinformatics.dev/getting_started/input_output.html#loading-bounding-boxes-tracks) for more details.

Note that when using `--save_frames`, the frames of the video are saved as-is, without added bounding boxes. The aim is to support the visualisation and correction of the predictions using the [VGG Image Annotator (VIA)](https://www.robots.ox.ac.uk/~vgg/software/via/) tool. To do so, follow the instructions of the [VIA Face track annotation tutorial](https://www.robots.ox.ac.uk/~vgg/software/via/docs/face_track_annotation.html).

If a file with ground-truth annotations is passed to the command (with the `--annotations_file` flag), the MOTA metric for evaluating tracking is computed and printed to screen.

<!-- When used in combination with the `--save_video` flag, the tracked video will contain predicted bounding boxes in red, and ground-truth bounding boxes in green. -- PR 216-->

To see the full list of possible arguments to the `evaluate-detector` command, run it with the `--help` flag.



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
