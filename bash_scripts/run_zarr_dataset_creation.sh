#!/bin/bash

#SBATCH -p gpu # partition (or gpu if needed)
#SBATCH -N 1   # number of nodes
#SBATCH --ntasks-per-node 2
#SBATCH --mem 64G
#SBATCH -t 0-20:00 # time (D-HH:MM)
#SBATCH -o slurm_array.%A-%a.%N.out
#SBATCH -e slurm_array.%A-%a.%N.err
#SBATCH --mail-type=ALL
#SBATCH --mail-user=s.minano@ucl.ac.uk
#SBATCH --array=0-26%9  # 27 videos, max 9 jobs at once

# NOTE: Adjust --array=0-N%M where:
# - N = number of rows in csv minus 1 (0-indexed)
# - M = max concurrent jobs

set -e
set -u
set -o pipefail

# ---------------------
# Define variables
# ----------------------
VIA_TRACKS_DIR="/ceph/zoo/users/sminano/loops_tracking_above_10th_percentile_slurm_1825237_2071125_2071084"
METADATA_CSV="/ceph/zoo/users/sminano/CrabsField/crab-loops/loop-frames-ffmpeg.csv"

ZARR_STORE_OUTPUT="/ceph/zoo/users/sminano/CrabTracks-slurm$SLURM_ARRAY_JOB_ID.zarr"
ZARR_MODE_STORE="a"    # a => append if store exists
ZARR_MODE_GROUP="w-"  # w- => throw error if writing to existing group

# location of SLURM logs
LOG_DIR=$ZARR_STORE_OUTPUT/logs
mkdir -p $LOG_DIR  # create if it doesnt exist

# Version of the codebase
GIT_BRANCH=smg/convert-to-zarr


# --------------------
# Check inputs
# --------------------
# Check number of video files in CSV matches max SLURM_ARRAY_TASK_COUNT
# if not, exit

# Get list of unique video files in csv (excluding header, column 3 is video_name)
LIST_VIDEOS=($(tail -n +2 "$METADATA_CSV" | cut -d',' -f3 | sort -u))
N_VIDEOS=${#LIST_VIDEOS[@]}

if [[ $SLURM_ARRAY_TASK_COUNT -ne $N_VIDEOS ]]; then
    echo "The number of array tasks does not match the number of videos in the input csv."
    echo "  Array tasks:    $SLURM_ARRAY_TASK_COUNT"
    echo "  Unique videos:  $N_VIDEOS"
    exit 1
fi

# -----------------------------
# Create virtual environment
# -----------------------------
# TODO: replace with uv
module load miniconda

ENV_NAME=crabs-zarr-$SLURM_ARRAY_JOB_ID-$SLURM_ARRAY_TASK_ID
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

# -------------------------
# Run extraction script
# -------------------------
VIDEO_NAME=${LIST_VIDEOS[$SLURM_ARRAY_TASK_ID]}
VIDEO_NAME_NO_EXT=${VIDEO_NAME%.mov} # remove .mov suffix
VIA_TRACKS_GLOB_PATTERN="$VIDEO_NAME_NO_EXT*.csv" # needs quotes

# Log arguments
echo "via_tracks_dir: $VIA_TRACKS_DIR"
echo "metadata_csv: $METADATA_CSV"
echo "zarr_store: $ZARR_STORE_OUTPUT"
echo "zarr_mode_store: $ZARR_MODE_STORE"
echo "zarr_mode_group: $ZARR_MODE_GROUP"
echo "via_tracks_glob_pattern: $VIA_TRACKS_GLOB_PATTERN"

# to time and log memory usage, prepend
# /usr/bin/time -v to the command
create-zarr-dataset  \
    --via_tracks_dir $VIA_TRACKS_DIR \
    --metadata_csv $METADATA_CSV \
    --zarr_store $ZARR_STORE_OUTPUT \
    --zarr_mode_store $ZARR_MODE_STORE \
    --zarr_mode_group $ZARR_MODE_GROUP \
    --via_tracks_glob_pattern "$VIA_TRACKS_GLOB_PATTERN"  # with quotes

# -----------------------------
# Cleanup
# ----------------------------
conda deactivate
conda remove --prefix $ENV_PREFIX --all -y

# ------------------
# Copy logs to LOG_DIR
# -------------------
mv slurm_array.$SLURM_ARRAY_JOB_ID-$SLURM_ARRAY_TASK_ID.$SLURMD_NODENAME.{err,out} $LOG_DIR

# make logs read only
chmod 444 $LOG_DIR/slurm_array.$SLURM_ARRAY_JOB_ID-$SLURM_ARRAY_TASK_ID.$SLURMD_NODENAME.{err,out}
