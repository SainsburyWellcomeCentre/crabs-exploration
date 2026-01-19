#!/bin/bash

#SBATCH -p gpu # partition (or gpu if needed)
#SBATCH -N 1   # number of nodes
#SBATCH --ntasks-per-node 2
#SBATCH --mem 8G
#SBATCH -t 0-20:00 # time (D-HH:MM)
#SBATCH -o slurm_extract.%A-%a.%N.out
#SBATCH -e slurm_extract.%A-%a.%N.err
#SBATCH --mail-type=ALL
#SBATCH --mail-user=s.minano@ucl.ac.uk
#SBATCH --array=0-233%20  # 234 rows with data in csv, max 20 jobs at once

# NOTE: Adjust --array=0-N%M where:
# - N = number of rows in csv minus 1 (0-indexed)
# - M = max concurrent jobs

set -e
set -u
set -o pipefail

# ---------------------
# Define variables
# ----------------------
CSV_PATH="/ceph/zoo/users/sminano/CrabsField/crab-loops/loop-frames-ffmpeg.csv"
INPUT_DIR="/ceph/zoo/raw/CrabField/ramalhete_2023"

OUTPUT_DIR="/ceph/zoo/processed/CrabField/ramalhete_2023/Loops"
mkdir -p $OUTPUT_DIR  # create if it doesnt exist

# location of SLURM logs
LOG_DIR=$OUTPUT_DIR/logs
mkdir -p $LOG_DIR  # create if it doesnt exist

# Version of the codebase
GIT_BRANCH=smg/extract-clips

# Whether to verify frame count after extracting the clips
VERIFY_FRAMES=true

# Fraction of frame duration to use as buffer around the PTS of the
# clip start and end frames. This is to ensure both frames are
# included in the output clip. PTS=timestamp for the start of the frame.
EPSILON_FRAME_FRACTION=0.25

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
    python=3.12

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

# ---------------------------------------
# Set flags based on boolean variables
# ---------------------------------------
if [ "$VERIFY_FRAMES" = "true" ]; then
    VERIFY_FRAMES_FLAG="--verify_frames"
else
    VERIFY_FRAMES_FLAG=""
fi


# -------------------------
# Run extraction script
# -------------------------
extract-loops \
    --csv_filepath $CSV_PATH \
    --input_dir $INPUT_DIR \
    --output_dir $OUTPUT_DIR \
    --epsilon_frame_fraction $EPSILON_FRAME_FRACTION \
    --slurm_array_task_id $SLURM_ARRAY_TASK_ID \
    $VERIFY_FRAMES_FLAG

echo "Completed extraction of clip with task ID = $SLURM_ARRAY_TASK_ID"
echo "--------------------------------------------------------"

# -----------------------------
# Cleanup
# ----------------------------
conda deactivate
conda remove --prefix $ENV_PREFIX --all -y

# ------------------
# Copy logs to LOG_DIR
# -------------------
mv slurm_extract.$SLURM_ARRAY_JOB_ID-$SLURM_ARRAY_TASK_ID.$SLURMD_NODENAME.{err,out} $LOG_DIR

# make logs read only
chmod 444 $LOG_DIR/slurm_extract.$SLURM_ARRAY_JOB_ID-$SLURM_ARRAY_TASK_ID.$SLURMD_NODENAME.{err,out}
