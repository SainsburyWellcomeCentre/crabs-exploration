#!/bin/bash

#SBATCH -p gpu # partition
#SBATCH --gres=gpu:1 # For any GPU: --gres=gpu:1. For a specific one: --gres=gpu:rtx5000
#SBATCH -N 1   # number of nodes
#SBATCH --ntasks-per-node 2
#SBATCH --mem 16G # memory pool for all cores
#SBATCH -t 5-00:00 # time (D-HH:MM);
#SBATCH -o slurm_array.%A-%a.%N.out
#SBATCH -e slurm_array.%A-%a.%N.err
#SBATCH --mail-type=ALL
#SBATCH --mail-user=s.minano@ucl.ac.uk
#SBATCH --array=0-26%9  # 27 videos, max 9 jobs at once

# NOTE: Adjust --array=0-N%M where:
# - N = number of video groups in the trajectories zarr store minus 1 (0-indexed)
# - M = max concurrent jobs

set -e
set -u
set -o pipefail

# ---------------------
# Define variables
# ----------------------
# Trajectories zarr store with the boxes to prompt SAM2 with,
# as written by create-zarr-dataset. One group per video.
BOXES_ZARR_STORE="/ceph/zoo/processed/CrabField/ramalhete_2023/CrabTracks/CrabTracks-slurm3644250.zarr"

# Directory with the clip videos the boxes refer to, named
# <video-id>-<clip-id>.mp4, as written by extract-loop-clips
CLIP_VIDEOS_DIR="/ceph/zoo/processed/CrabField/ramalhete_2023/Loops"

# Path to the mask config file
MASK_CONFIG_FILE="/ceph/zoo/users/sminano/cluster_mask_config.yaml"

# Output store. Every job in the array appends its own video group to it,
# so the name must not depend on the task. Naming it after the array job ID
# ensures different runs generate different stores.
MASK_ZARR_STORE_OUTPUT="/ceph/zoo/users/sminano/CrabMasks-slurm$SLURM_ARRAY_JOB_ID.zarr"
ZARR_MODE_STORE="a"   # a => append if store exists
ZARR_MODE_GROUP="w-"  # w- => throw error if writing to existing group

# location of SLURM logs
LOG_DIR=$MASK_ZARR_STORE_OUTPUT/logs
mkdir -p $LOG_DIR  # create if it doesnt exist

# Version of the codebase
GIT_BRANCH=smg/mask-tracked-crabs

# --------------------
# Check inputs
# --------------------
# Check the number of video groups in the input store matches
# max SLURM_ARRAY_TASK_COUNT; if not, exit.
# A video group is a subdirectory of the store, excluding our own log dir.
mapfile -t LIST_VIDEOS < <(
    find "$BOXES_ZARR_STORE" -mindepth 1 -maxdepth 1 -type d \
        -not -name "logs*" -exec basename {} \; | sort
)
N_VIDEOS=${#LIST_VIDEOS[@]}

if [[ $SLURM_ARRAY_TASK_COUNT -ne $N_VIDEOS ]]; then
    echo "The number of array tasks does not match the number of video groups in the input store."
    echo "  Array tasks:   $SLURM_ARRAY_TASK_COUNT"
    echo "  Video groups:  $N_VIDEOS"
    exit 1
fi

# ---------------------------
# Create virtual environment
# ---------------------------
# We create a virtual environment for each job in the array,
# under tmpdir, using uv. We create a separate environment per job
# to avoid all jobs to race and run uv venv and uv pip install simultaneously,
# which could cause issues.
module load uv

# set uv cache dir to /ceph/scratch/sminano
# (should be faster than /nfs/nhome/live/sminano/.cache/uv and
# gets purged regularly)
export UV_CACHE_DIR=/ceph/python_envs/sminano/uv-cache

# uv holds a lock on the shared cache while it fetches and builds a git
# dependency (SAM2 below). With several array jobs starting together on a
# slow shared filesystem, the default 300 s wait for that lock is not enough
# and the other jobs fail with "Timeout when waiting for lock".
export UV_LOCK_TIMEOUT=3600  # seconds

ENV_NAME=crabs-mask-$SLURM_ARRAY_JOB_ID-$SLURM_ARRAY_TASK_ID
ENV_PREFIX=/ceph/python_envs/sminano/$ENV_NAME

# create virtual environment with uv
uv venv $ENV_PREFIX --python 3.12

# activate environment
source $ENV_PREFIX/bin/activate

# install crabs package in virtual env. This brings in torch, which SAM2
# builds against below.
CRABS_URL="git+https://github.com/SainsburyWellcomeCentre/crabs-exploration.git@$GIT_BRANCH"
uv pip install "crabs @ $CRABS_URL"

# We install sam2 without build isolation. This is so that it picks up the torch
# already installed in the environment. We also explicitly install setuptools,
# because SAM2 needs it (it is belt-and-braces tho, because torch depends on
# setuptools today, but that may change).
uv pip install setuptools
uv pip install --no-build-isolation-package sam-2 \
    "crabs[masks] @ $CRABS_URL"

# cache the SAM2 checkpoint on ceph rather than in the home directory,
# so that it is downloaded once and shared across jobs
export HF_HOME=/ceph/scratch/sminano/huggingface

# log pip and python locations
echo $ENV_PREFIX
which python
which pip

# print the version of crabs package (last number is the commit hash)
echo "Git branch: $GIT_BRANCH"
uv pip show crabs
echo "-----"

# ------------------------------------
# GPU specs
# ------------------------------------
echo "Memory used per GPU before masking"
echo $(nvidia-smi --query-gpu=name,memory.total,memory.free,memory.used --format=csv) #noheader
echo "-----"

# -------------------------
# Run masking script
# -------------------------
# video group masked in this job
VIDEO_ID=${LIST_VIDEOS[$SLURM_ARRAY_TASK_ID]}

# Log arguments
echo "boxes: $BOXES_ZARR_STORE"
echo "videos: $CLIP_VIDEOS_DIR"
echo "output_store: $MASK_ZARR_STORE_OUTPUT"
echo "match: $VIDEO_ID"
echo "mask_config_file: $MASK_CONFIG_FILE"

# --match is an exact video group name here, so this job masks every clip
# of one video. --output_store names the store rather than letting the
# command pick a timestamped name, so all jobs write into the same one.
mask-tracked-crabs  \
    --boxes $BOXES_ZARR_STORE  \
    --videos $CLIP_VIDEOS_DIR  \
    --output_store $MASK_ZARR_STORE_OUTPUT  \
    --match "$VIDEO_ID"  \
    --mask_config_file $MASK_CONFIG_FILE  \
    --zarr_mode_store $ZARR_MODE_STORE  \
    --zarr_mode_group $ZARR_MODE_GROUP  \
    --accelerator gpu

echo "Completed masking of video $VIDEO_ID with task ID = $SLURM_ARRAY_TASK_ID"
echo "--------------------------------------------------------"

# -------------------------------------------
# Copy mask config to output store
# -------------------------------------------
cp "$MASK_CONFIG_FILE" "$MASK_ZARR_STORE_OUTPUT"/"$VIDEO_ID"_mask_config.yaml

echo "Copied $MASK_CONFIG_FILE to $MASK_ZARR_STORE_OUTPUT"

# ------------------
# Copy logs to LOG_DIR
# -------------------
mv slurm_array.$SLURM_ARRAY_JOB_ID-$SLURM_ARRAY_TASK_ID.$SLURMD_NODENAME.{err,out} $LOG_DIR

# make logs read only
chmod 444 $LOG_DIR/slurm_array.$SLURM_ARRAY_JOB_ID-$SLURM_ARRAY_TASK_ID.$SLURMD_NODENAME.{err,out}

# -----------------------------
# Cleanup
# ----------------------------
deactivate
rm -rf $ENV_PREFIX
