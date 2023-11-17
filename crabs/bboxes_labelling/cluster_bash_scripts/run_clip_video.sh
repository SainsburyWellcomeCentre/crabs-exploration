#!/bin/bash

#SBATCH -p gpu # partition
#SBATCH -N 1   # number of nodes
#SBATCH --mem 8G # memory pool for all cores
#SBATCH -n 2 # number of cores
#SBATCH -t 03-00:00 # time (D-HH:MM)
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
INPUT_DIR=/ceph/zoo/raw/CrabField/ramalhete_2023/09.08.2023-Day2/09.08.2023-04-Right.MOV
OUTPUT_DIR=/ceph/scratch/nikkna/event_cut/

# script location
SCRATCH_PERSONAL_DIR=/ceph/scratch/nikkna
SCRIPT_DIR=$SCRATCH_PERSONAL_DIR/crabs-exploration/"bboxes labelling"

# TODO: set NUMEXPR_MAX_THREADS?
# NumExpr detected 40 cores but "NUMEXPR_MAX_THREADS" not set,
# so enforcing safe limit of 8.

# -------------------
# Run python script
# -------------------
python "$SCRIPT_DIR"/clip_video.py  \
 --video_path $INPUT_DIR \
 --out_path $OUTPUT_DIR \
