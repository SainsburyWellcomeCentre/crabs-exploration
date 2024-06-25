### Running Locally

[TODO: expand these instructions]

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
