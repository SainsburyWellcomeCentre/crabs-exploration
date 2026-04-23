# %%%%%%%%%%%%%%%%%%%%%%%
# Generate YOLO detection training data via OCTRON

from octron.yolo_octron.yolo_octron import YOLO_octron

yolo = YOLO_octron(project_path=OUTPUT_DIR, clean_training_dir=True)
yolo.train_mode = "segment"

# Add label dict
yolo.label_dict = {
    output_masks_dir.as_posix(): {
        # label description
        # key is class_ID
        0: {
            "label": LABEL_NAME,  # class name for YOLO
            "original_id": 1,  # from napari
            "frames": mask_array.attrs["annotated_frames"],
            "masks": [mask_array],
            "color": [1.0, 0.0, 0.0, 1.0],  # for GUI only
        },
        # session metadata
        "video": image_array,
        "video_file_path": IMAGES_DIR,
    }
}

# %%
# Extract polygons from masks
# Check the output: each label
# entry in label_dict should now have a "polygons" key.

# it's a generator so we need to consume it
# for it to run
yolo.enable_watershed = False
for _ in yolo.prepare_polygons():
    pass

# %%
# Train / val/ test split
yolo.prepare_split(
    training_fraction=0.8,
    validation_fraction=0.2,
    verbose=True,
)

# %%
# Write images and YOLO label file
for _ in yolo.create_training_data_segment():
    pass

# # Replace copied PNGs with symlinks to originals
# for split_dir in yolo.data_path.iterdir():
#     for img_file in split_dir.glob("*.png"):
#         # Would the frame_id actually work?
#         frame_id = int(img_file.stem.split("_")[-1])
#         original = png_files[frame_id]
#         img_file.unlink()
#         img_file.symlink_to(original)

# %%
# Write config
yolo.write_yolo_config(train_mode="segment")

# - yolo.data_path has the YOLO segmentation dataset
# - yolo.config_path has the data.yaml
# %%
# Training

segmentor_model = YOLO("yolo11n-seg.pt")
segmentor_model.train(
    data=str(yolo.config_path),  # path to data.yaml
    epochs=100,
    imgsz=1280,  # 1600? 2144?
)
# %%
# Inference via OCTRON, or Sahi?


# Sahi:
# from sahi import AutoDetectionModel
# from sahi.predict import get_sliced_prediction

# model = AutoDetectionModel.from_pretrained(
#     model_type="ultralytics",
#     model_path="path/to/best.pt",
#     confidence_threshold=0.5,
#     device="mps",
# )

# result = get_sliced_prediction(
#     image="path/to/frame.png",
#     detection_model=model,
#     slice_height=1024,
#     slice_width=1024,
#     overlap_height_ratio=0.2,
#     overlap_width_ratio=0.2,
# )
