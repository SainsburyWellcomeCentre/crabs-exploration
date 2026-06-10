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
CSV_PATH="/ceph/zoo/users/sminano/CrabLabels/crab-loops/loop-frames-ffmpeg.csv"
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
# Set up uv
# -----------------------------
# extract_loop_clips.py is a standalone (PEP 723) script: uv fetches it
# from the repository, resolves its inline dependencies into an ephemeral
# environment, and runs it. No package install or virtual environment is
# needed.
module load uv

# set uv cache dir to /ceph/scratch/sminano
# (should be faster than the home directory cache and gets purged regularly)
export UV_CACHE_DIR=/ceph/scratch/sminano/uv-cache
# copy (instead of symlink) files across filesystems (ceph cache vs tmpfs)
export UV_LINK_MODE=copy
export UV_HTTP_TIMEOUT=120  # seconds

# Remote URL of the standalone script for the selected branch
SCRIPT_URL="https://raw.githubusercontent.com/SainsburyWellcomeCentre/crabs-exploration/$GIT_BRANCH/scripts/extract_loop_clips.py"

echo "Git branch: $GIT_BRANCH"
echo "Script: $SCRIPT_URL"
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
uv run "$SCRIPT_URL" \
    --csv_filepath $CSV_PATH \
    --input_dir $INPUT_DIR \
    --output_dir $OUTPUT_DIR \
    --epsilon_frame_fraction $EPSILON_FRAME_FRACTION \
    --slurm_array_task_id $SLURM_ARRAY_TASK_ID \
    $VERIFY_FRAMES_FLAG

echo "Completed extraction of clip with task ID = $SLURM_ARRAY_TASK_ID"
echo "--------------------------------------------------------"

# ------------------
# Copy logs to LOG_DIR
# -------------------
mv slurm_extract.$SLURM_ARRAY_JOB_ID-$SLURM_ARRAY_TASK_ID.$SLURMD_NODENAME.{err,out} $LOG_DIR

# make logs read only
chmod 444 $LOG_DIR/slurm_extract.$SLURM_ARRAY_JOB_ID-$SLURM_ARRAY_TASK_ID.$SLURMD_NODENAME.{err,out}
