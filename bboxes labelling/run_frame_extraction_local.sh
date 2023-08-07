#!/bin/bash

CRABS_REPO_DIR=/Users/sofia/Documents_local/project_Zoo_crabs/crabs-exploration
OUTPUT_DIR=$CRABS_REPO_DIR/data/crab_sample_data/sample_frames

DATA_DIR=/Volumes/zoo/raw/CrabField/ramalhete_2021

# -------------------
# Run python script
# -------------------
python $CRABS_REPO_DIR"/bboxes labelling/extract_frames_to_label_w_sleap.py" \
 $DATA_DIR/camera2/NINJAV_S001_S001_T001.MOV \
 --output_path $OUTPUT_DIR \
 --video_extensions MOV \
 --initial_samples 300 \
 --scale 0.75 \
 --n_components 5 \
 --n_clusters 5 \
 --per_cluster 8 \

# python $CRABS_REPO_DIR"/bboxes labelling/extract_frames_to_label_w_sleap.py" \
#  $DATA_DIR/camera2/NINJAV_S001_S001_T001.MOV \
#  --output_path $OUTPUT_DIR \
#  --video_extensions MOV \
#  --initial_samples 10 \
#  --scale 0.5 \
#  --n_components 5 \
#  --n_clusters 5 \
#  --per_cluster 2 \
#  --compute_features_per_video

# python $CRABS_REPO_DIR"/bboxes labelling/extract_frames_to_label_w_sleap.py" \
#  $CRABS_REPO_DIR/data/crab_sample_data/sample_clips \
#  $CRABS_REPO_DIR/data/cam2_NINJAV_S001_S001_T004_out_domain_sample.mp4 \
#  --output_path $OUTPUT_DIR \
#  --video_extensions avi mp4 \
#  --initial_samples 300 \
#  --scale 0.75 \
#  --n_components 5 \
#  --n_clusters 5 \
#  --per_cluster 8 \
#  --compute_features_per_video
