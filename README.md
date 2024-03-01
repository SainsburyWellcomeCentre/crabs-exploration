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

### Running Locally

For training

```bash
python crabs/detection_tracking/train_model.py --main_dir {parent_directory_of_frames_and_annotation} --annotation_file {annotation_file.json}
```

For evaluation

```bash
python crabs/detection_tracking/evaluate_model.py --model_dir {directory_to_saved_model} --main_dir {parent_directory_of_frames_and_annotation} --annotation_file {annotation_file.json}
```

For tracking

```bash
python crabs/detection_tracking/inference_model.py --model_dir {directory_to_saved_model} --vid_dir {parent_directory_of_a_video}
```
