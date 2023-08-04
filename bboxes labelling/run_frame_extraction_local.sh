#!/bin/bash

CRABS_REPO_DIR=/Users/sofia/Documents_local/project_Zoo_crabs/crabs-exploration
OUTPUT_DIR=$CRABS_REPO_DIR/data/crab_sample_data/sample_frames

# -------------------
# Run python script
# -------------------
python $CRABS_REPO_DIR"/bboxes labelling/extract_frames_to_label_w_sleap.py" \
 $CRABS_REPO_DIR/data/crab_sample_data/sample_clips \
 $CRABS_REPO_DIR/data/cam2_NINJAV_S001_S001_T004_out_domain_sample.mp4 \
 --output_path $OUTPUT_DIR \
 --video_extensions avi mp4 \
 --initial_samples 10 \
 --scale 1.0 \
 --n_components 3 \
 --n_clusters 5 \
 --per_cluster 2 \
 --compute_features_per_video
