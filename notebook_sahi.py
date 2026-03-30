"""From https://colab.research.google.com/github/obss/sahi/blob/main/demo/inference_for_torchvision.ipynb"""

# %%
from pathlib import Path

import numpy as np
from IPython.display import Image as IPythonImage
from PIL import Image
from sahi import AutoDetectionModel
from sahi.predict import get_prediction, get_sliced_prediction, predict
from sahi.scripts.coco_error_analysis import analyse
from sahi.scripts.coco_evaluation import evaluate

from crabs.detector.models import FasterRCNN

# %%
# Input data
model_ckpt_path = Path(
    "/home/sminano/swc/project_crabs/ml-runs/617393114420881798/4acc37206b1e4f679d535c837bee2c2f/checkpoints/last.ckpt"
)


aug2023_annotations = Path(
    "/home/sminano/swc/project_crabs/data/aug2023-full/annotations/VIA_JSON_combined_coco_gen.json"
)

aug2023_images_dir = Path(
    "/home/sminano/swc/project_crabs/data/aug2023-full/frames/"
)

# %%
# Helper


class ImageLazyArray:
    def __init__(self, images_dir: str | Path, extension: str = "png"):
        """Set images paths and shape."""
        self.imgs_paths = sorted(Path(images_dir).glob(f"*.{extension}"))

        # All iamges assumed same shape as the first
        sample_img = np.asarray(Image.open(self.imgs_paths[0]))
        self.img_h, self.img_w, self.img_c = sample_img.shape

    def __getitem__(self, idx):
        return np.asarray(Image.open(self.imgs_paths[idx]))

    def __len__(self):
        return len(self.imgs_paths)

    @property
    def shape(self):
        return (
            len(self.imgs_paths),
            self.img_h,
            self.img_w,
            self.img_c,
        )  # B, H, W, C


# %%
# Load model from checkpoint
# Load the Lightning module from checkpoint
lightning_module = FasterRCNN.load_from_checkpoint(model_ckpt_path)
lightning_module.eval()

detection_model = AutoDetectionModel.from_pretrained(
    model_type="torchvision",
    # model_path = model_ckpt_path, # ATT: not a valid torchvision model!
    model=lightning_module.model,  # Extract the underlying torchvision model
    category_mapping={"1": "crab"},
    confidence_threshold=0.0,  # predictions with score < threshold are discarded
    image_size=None,  # Inference input size. None matches training conditions
    device="cuda:0",
    load_at_init=True,
)

print(detection_model.device)

# %%
# Get Aug dataset for evaluation

aug_array = ImageLazyArray(images_dir=aug2023_images_dir)

print(aug_array.shape)

# %%
# Select sample image
input_image = aug_array[0]

# %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
# Option 1: inference on single image with no tiling
result = get_prediction(input_image, detection_model)


# %%
# Export visuals and inspect
export_dir_not_sliced = Path("demo_data/not-sliced")
result.export_visuals(export_dir=export_dir_not_sliced)

# %%
IPythonImage(export_dir_not_sliced / "prediction_visual.png")


# %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
# Option 2: sliced inference on single image
result = get_sliced_prediction(
    input_image,
    detection_model,
    slice_height=2160,
    slice_width=2160,
    overlap_height_ratio=0.1,
    overlap_width_ratio=0.1,
)

# %%
# Export visuals and inspect
export_dir_sliced = Path("demo_data/sliced/")
result.export_visuals(export_dir=export_dir_sliced)

IPythonImage(export_dir_sliced / "prediction_visual.png")


# %%
# Inspect predictions
# Predictions are returned as sahi.prediction.PredictionResult,
preds_list = result.object_prediction_list

# Convert to COCO (ok? image_id=None?)
pred_per_img_list_coco = result.to_coco_annotations()
# result.to_coco_predictions(image_id=1)[:3] ---> first 3 annotations, image_id=1


# %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
# Batch prediction standard only OR standard+sliced

# Notes
# no_standard_prediction: bool
#     Dont perform standard prediction. Default: False.
# no_sliced_prediction: bool
#     Dont perform sliced prediction. Default: False.
# novisual
#     Dont export predicted video/image visuals.
# dataset_json_path: str
#     If coco file path is provided, detection results will
#     be exported in coco json format. (ground-truth)
# project: str
#     Save results to project/name.
# name: str
#     Save results to project/name.
# visual_hide_labels: bool, optional
#     If True, class label names won't be shown on the exported visuals.

# visual_bbox_thickness: int | None = None,
# visual_text_size: float | None = None,
# visual_text_thickness: int | None = None,
# visual_hide_labels: bool = False,
# visual_hide_conf: bool = False,
# visual_export_format: str = "png",
# %%
flag_standard_only = False
flag_save_visuals = True

predict(
    detection_model=detection_model,
    dataset_json_path=str(aug2023_annotations),  # ground-truth COCO JSON
    source=aug2023_images_dir,  # used as base directory prepended to img filenames in COCO
    slice_height=2160,
    slice_width=2160,
    overlap_height_ratio=0.1,
    overlap_width_ratio=0.1,
    novisual=not flag_save_visuals,
    no_sliced_prediction=flag_standard_only,
    visual_bbox_thickness=3,
    visual_hide_labels=False,  # ----> hide labels also hides confidence?
    visual_hide_conf=False,
    visual_text_thickness=3,
    visual_text_size=2,  # ---------------------->
    project="sahi_predictions/aug2023",
    name=("standard" if flag_standard_only else "sliced"),
)

# %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
# Evaluate in full Aug

# Evaluate standard
evaluate(
    dataset_json_path=str(aug2023_annotations),
    result_json_path="sahi_predictions/aug2023/standard/result.json",
    out_dir="sahi_predictions/aug2023/standard/",  # dir to save eval result
)

#  Average Precision  (AP) @[ IoU=0.50:0.95 | area=   all | maxDets=500 ] = 0.191
#  Average Precision  (AP) @[ IoU=0.50      | area=   all | maxDets=500 ] = 0.555
#  Average Precision  (AP) @[ IoU=0.75      | area=   all | maxDets=500 ] = 0.073
#  Average Precision  (AP) @[ IoU=0.50      | area= small | maxDets=500 ] = 0.196
#  Average Precision  (AP) @[ IoU=0.50      | area=medium | maxDets=500 ] = 0.751
#  Average Precision  (AP) @[ IoU=0.50      | area= large | maxDets=500 ] = 0.000
#  Average Precision  (AP) @[ IoU=0.50:0.95 | area= small | maxDets=500 ] = 0.047
#  Average Precision  (AP) @[ IoU=0.50:0.95 | area=medium | maxDets=500 ] = 0.269
#  Average Precision  (AP) @[ IoU=0.50:0.95 | area= large | maxDets=500 ] = 0.000
#  Average Recall     (AR) @[ IoU=0.50:0.95 | area= small | maxDets=500 ] = 0.045
#  Average Recall     (AR) @[ IoU=0.50:0.95 | area=medium | maxDets=500 ] = 0.363
#  Average Recall     (AR) @[ IoU=0.50:0.95 | area= large | maxDets=500 ] = 0.000
# %%
# Evaluate sliced
evaluate(
    dataset_json_path=str(aug2023_annotations),
    result_json_path="sahi_predictions/aug2023/sliced/result.json",
    out_dir="sahi_predictions/aug2023/sliced",
)


# CLI:
# sahi coco evaluate
#   --dataset_json_path "/home/sminano/swc/project_crabs/data/aug2023-full/annotations/VIA_JSON_combined_coco_gen.json"
#   --result_json_path "sahi_predictions/aug2023/sliced/result.json"
#
#  Average Precision  (AP) @[ IoU=0.50:0.95 | area=   all | maxDets=500 ] = 0.196
#  Average Precision  (AP) @[ IoU=0.50      | area=   all | maxDets=500 ] = 0.602
#  Average Precision  (AP) @[ IoU=0.75      | area=   all | maxDets=500 ] = 0.065
#  Average Precision  (AP) @[ IoU=0.50      | area= small | maxDets=500 ] = 0.293
#  Average Precision  (AP) @[ IoU=0.50      | area=medium | maxDets=500 ] = 0.781
#  Average Precision  (AP) @[ IoU=0.50      | area= large | maxDets=500 ] = 0.000
#  Average Precision  (AP) @[ IoU=0.50:0.95 | area= small | maxDets=500 ] = 0.068
#  Average Precision  (AP) @[ IoU=0.50:0.95 | area=medium | maxDets=500 ] = 0.266
#  Average Precision  (AP) @[ IoU=0.50:0.95 | area= large | maxDets=500 ] = 0.000
#  Average Recall     (AR) @[ IoU=0.50:0.95 | area= small | maxDets=500 ] = 0.079
#  Average Recall     (AR) @[ IoU=0.50:0.95 | area=medium | maxDets=500 ] = 0.375
#  Average Recall     (AR) @[ IoU=0.50:0.95 | area= large | maxDets=500 ] = 0.000

# %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
# Analyse standard only OR standard + slice

# Modify GT to include background as catID = 0
aug2023_annotations_with_catID0 = aug2023_annotations.parent / (
    aug2023_annotations.stem + "_catID0included.json"
)

select_run = "standard"  # or sliced

# Run error analysis
analyse(
    dataset_json_path=str(aug2023_annotations_with_catID0),
    result_json_path=f"sahi_predictions/aug2023/{select_run}/result.json",
    out_dir=f"sahi_predictions/aug2023/{select_run}",
)

# - The Loc curve is a standard COCO error analysis concept where the
#   IoU threshold is relaxed to 0.1 (any overlap > 0.1 counts as a correct
#   localisation).
# - The curves correspond to a fixed IOU threshold, and moving along the
#   curve changes the confidence threshold
# - Small objects — if crabs are small in the image, even a few pixels of
#   box error can push IoU below 0.5.
# # %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
# # Analyse sliced

# analyse(
#     dataset_json_path=str(aug2023_annotations_with_catID0),
#     result_json_path="sahi_predictions/aug2023/sliced/result.json",
#     out_dir="sahi_predictions/aug2023/sliced"
# )


# %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
# Compute true positive, false positives, missed detections

# coco_eval.params.iouThrs is [0.50, 0.55, ..., 0.95] by default, so iou_idx=0 gives IoU=0.50
# This sums across all images and all categories (just "crab" here)
# The gt_matches shape is (n_iou_thrs, n_gt) — same indexing as dt_matches
# This uses the default maxDets=100; if you want 500 like in your evaluate() calls, add coco_eval.params.maxDets = [500] before coco_eval.evaluate()

import json

from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval

run_setting = "standard"
IOU_threshold = 0.1
conf_threshold = 0.5

# Load GT
coco_gt = COCO(str(aug2023_annotations))

# Load predictions
# coco_dt = coco_gt.loadRes(f"sahi_predictions/aug2023/{run_setting}/result.json")
with open(f"sahi_predictions/aug2023/{run_setting}/result.json") as f:
    preds = json.load(f)
preds_filtered = [p for p in preds if p["score"] >= conf_threshold]
coco_dt = coco_gt.loadRes(preds_filtered)


# Run evaluation
coco_eval = COCOeval(coco_gt, coco_dt, iouType="bbox")

# Set IOU threshold
# default: iouThrs = [0.5, 0.55, ..., 0.95]
coco_eval.params.iouThrs = np.array([IOU_threshold])

# Set limits for GT ignore
coco_eval.params.areaRng = [[0, 1e10]]
coco_eval.params.areaRngLbl = ["all"]
coco_eval.params.maxDets = [500]

coco_eval.evaluate()

# Extract TP, FP, FN at predefined IoU
iou_idx = 0  # index for iouThrs

tp, fp, fn = 0, 0, 0
for eval_img in coco_eval.evalImgs:
    if eval_img is None:
        continue
    dt_matches = eval_img["dtMatches"][
        iou_idx
    ]  # matched detection for each det
    dt_ignore = eval_img["dtIgnore"][iou_idx]  # ignored detections
    gt_ignore = np.array(eval_img["gtIgnore"])

    # TP: detections matched to a GT and not ignored
    tp += int(np.sum((dt_matches > 0) & ~dt_ignore))

    # FP: detections not matched and not ignored
    fp += int(np.sum((dt_matches == 0) & ~dt_ignore))

    # FN: GT annotations not matched and not ignored
    # (missed detections)
    gt_matched = np.array(eval_img["gtMatches"][iou_idx])
    fn += int(np.sum((gt_matched == 0) & ~gt_ignore))

print("---------------------------------")
print(f"Run setting: {run_setting}")
print(f"IoU threshold: {IOU_threshold}")
print(f"Conf threshold: {conf_threshold}")
print(f"TP: {tp}, FP: {fp}, FN: {fn}")
print(f"Precision: {tp / (tp + fp):.3f}")
print(f"Recall:    {tp / (tp + fn):.3f}")


total_gt_ignored = sum(
    int(np.sum(eval_img["gtIgnore"]))
    for eval_img in coco_eval.evalImgs
    if eval_img is not None
)
print(f"Total ignored GT: {total_gt_ignored}")
# Should be 0


# standard IOU = 0.5, conf_th = 0.0
# TP: 22098, FP: 10389, FN: 14508
# Precision: 0.680
# Recall:    0.604

# standard IOU = 0.1,
# conf_th = 0.0
# TP: 26749, FP: 5738, FN: 9857
# Precision: 0.823
# Recall:    0.731
#
# Conf threshold: 0.5
# TP: 26230, FP: 3712, FN: 10376
# Precision: 0.876
# Recall:    0.717

# ----------------
# sliced IOU = 0.5,
# conf_th = 0.0
# TP: 24941, FP: 24564, FN: 11665
# Precision: 0.504
# Recall:    0.681


# sliced IOU = 0.1,
# conf_th = 0.0
# TP: 31499, FP: 18006, FN: 5107
# Precision: 0.636
# Recall:    0.860
#
# Conf threshold: 0.5
# TP: 30753, FP: 9885, FN: 5853
# Precision: 0.757
# Recall:    0.840

# %%
