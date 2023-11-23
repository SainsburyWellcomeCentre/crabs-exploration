#!/bin/bash

#SBATCH -p gpu # partition
#SBATCH -N 1   # number of nodes
#SBATCH --mem 8G # memory pool for all cores
#SBATCH -n 2 # number of cores
#SBATCH -t 3-00:00 # time (D-HH:MM)
#SBATCH --mail-type=ALL
#SBATCH --mail-user=n.aznan@ucl.ac.uk

# ---------------------
# Load required modules
# ----------------------
module load SLEAP

# ---------------------
# Define environment variables
# ----------------------
# input/output dirs
INPUT_DIR=/ceph/zoo/users/sminano/crabs_bboxes_labels/20230816_ramalhete2023_day2_combined/extracted_frames.json
OUTPUT_DIR=/ceph/zoo/users/nikkna/event_clips/stacked_images/

# script location
SCRATCH_PERSONAL_DIR=/ceph/scratch/nikkna
SCRIPT_DIR=$SCRATCH_PERSONAL_DIR/crabs-exploration/"bboxes labelling"

# TODO: set NUMEXPR_MAX_THREADS?
# NumExpr detected 40 cores but "NUMEXPR_MAX_THREADS" not set,
# so enforcing safe limit of 8.

# -------------------
# Run python script
# -------------------
python "$SCRIPT_DIR"/additional_channels_extraction.py  \
 --json_path $INPUT_DIR \
 --out_dir $OUTPUT_DIR \;Q:q
