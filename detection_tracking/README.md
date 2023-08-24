## About

`detection_tracking` is used to train detection model on the dataset, run the test and the inference with the tracking object.

## Data structue

    .
    ├── ...
    ├── dataset                    
    │   ├── images
    │   │   ├── train
    │   │   ├── test    
    │   ├── labels
    │   │   ├── train.json
    │   │   ├── test.json       
    └── ...

### Requirement

```pip install -r requirements.txt``` 

### How to use

To run the training, run: 
``` 
python train_model.py --main_dir <../dataset>
```

To run the test, run:

```
python test_model.py --model_dir <model.pt> --main_dir <../dataset>
```

To run the inference, run:

```
python inference_model.py --model_dir <model.pt> --vid_dir <./video.mp4>
```