#!/bin/bash

#SBATCH -p cpu # partition (or gpu if needed)
#SBATCH -N 1   # number of nodes
#SBATCH --ntasks-per-node 2
#SBATCH --mem 8G
#SBATCH -t 0-12:00 # time (D-HH:MM)
#SBATCH -o slurm_extract.%A-%a.%N.out
#SBATCH -e slurm_extract.%A-%a.%N.err
#SBATCH --mail-type=ALL
#SBATCH --mail-user=s.minano@ucl.ac.uk
#SBATCH --array=0-234%10  # 235 rows in CSV, max 10 jobs at once

# NOTE: Adjust --array=0-N%M where:
# - N = number of rows in CSV minus 1 (0-indexed)
# - M = max concurrent jobs

set -e
set -u
set -o pipefail

# ---------------------
# Define variables
# ----------------------
CSV_PATH="/ceph/zoo/users/sminano/CrabsField/crab-loops/loop-frames-ffmpeg.csv"
INPUT_DIR="/ceph/zoo/users/sminano/crabs_input_videos_sample"  # /ceph/zoo/raw/CrabField/ramalhete_2023

OUTPUT_DIR="/ceph/zoo/users/sminano/crab_loops_clips"
mkdir -p $OUTPUT_DIR  # create if it doesnt exist

# location of SLURM logs
LOG_DIR=$OUTPUT_DIR/logs
mkdir -p $LOG_DIR  # create if it doesnt exist

# Version of the codebase
# TODO: point to branch with extract_clips script
GIT_BRANCH=main

# Python script location
# TODO: make an entrypoint?
SCRIPT_PATH="/ceph/zoo/users/sminano/crabs-exploration/scripts/extract_clips.py"


# --------------------
# Check inputs
# --------------------
# Check number of rows in CSV matches max SLURM_ARRAY_TASK_COUNT
# if not, exit

# Count number of rows in CSV (excluding header)
# tail -n +2 skips the header line, grep counts non-empty lines
NUM_CSV_ROWS=$(tail -n +2 "$CSV_PATH" | grep -c "^")

if [[ $SLURM_ARRAY_TASK_COUNT -ne $NUM_CSV_ROWS ]]; then
    echo "The number of array tasks does not match the number of rows in the input csv. "
    echo "  Array tasks: $SLURM_ARRAY_TASK_COUNT"
    echo "  CSV rows:    $NUM_CSV_ROWS"
    exit 1
fi

# -----------------------------
# Create virtual environment
# -----------------------------
# TODO: replace with uv
module load miniconda

ENV_NAME=crabs-extract-$SLURM_ARRAY_JOB_ID-$SLURM_ARRAY_TASK_ID
ENV_PREFIX=$TMPDIR/$ENV_NAME

conda create \
    --prefix $ENV_PREFIX \
    -y \
    python=3.10

# activate environment
source activate $ENV_PREFIX

# install crabs package in virtual env
python -m pip install git+https://github.com/SainsburyWellcomeCentre/crabs-exploration.git@$GIT_BRANCH

# log pip and python locations
echo $ENV_PREFIX
which python
which pip

# print the version of crabs package (last number is the commit hash)
echo "Git branch: $GIT_BRANCH"
conda list crabs
echo "-----"


# -------------------------
# Run extraction script
# -------------------------
python $SCRIPT_PATH \
    --csv_path $CSV_PATH \
    --input_dir $INPUT_DIR \
    --output_dir $OUTPUT_DIR \
    --array_task_id $SLURM_ARRAY_TASK_ID \
    --verify_frames

echo "Completed extraction of clip number $SLURM_ARRAY_TASK_ID"
echo "--------------------------------------------------------"

# ------------------
# Copy logs to LOG_DIR
# -------------------
mv slurm_array.$SLURMD_NODENAME.$SLURM_ARRAY_JOB_ID-$SLURM_ARRAY_TASK_ID.{err,out} /$LOG_DIR

# -----------------------------
# Cleanup
# ----------------------------
conda deactivate
conda remove --prefix $ENV_PREFIX --all -y