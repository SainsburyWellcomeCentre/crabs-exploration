#!/bin/bash

#SBATCH -p gpu # a100 # partition
#SBATCH --gres=gpu:1
#SBATCH -N 1   # number of nodes
#SBATCH --ntasks-per-node 8 # 2 # max number of tasks per node
#SBATCH --mem 64G # memory pool for all cores
#SBATCH -t 3-00:00 # time (D-HH:MM)
#SBATCH -o slurm.%A.%N.out
#SBATCH -e slurm.%A.%N.err
#SBATCH --mail-type=ALL
#SBATCH --mail-user=s.minano@ucl.ac.uk

# ---------------------
# Source bashrc
# ----------------------
# Otherwise `which python` points to the miniconda module's Python
source ~/.bashrc

# memory
# see https://pytorch.org/docs/stable/notes/cuda.html#environment-variables
PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

# -----------------------------
# Error settings for bash
# -----------------------------
# see https://wizardzines.com/comics/bash-errors/
set -e  # do not continue after errors
set -u  # throw error if variable is unset
set -o pipefail  # make the pipe fail if any part of it fails

# ---------------------
# Define variables
# ----------------------

# video and inference config
VIDEO_DIR=/ceph/zoo/raw/CrabField/ramalhete_2023/Escapes
CONFIG_FILE=/ceph/zoo/users/sminano/cluster_tracking_config.yaml

# checkpoint
TRAINED_MODEL_PATH=/ceph/zoo/users/sminano/ml-runs-all/ml_runs-nikkna-copy/243676951438603508/8dbe61069f17453a87c27b4f61f6e681/checkpoints/last.ckpt

# output directory
OUTPUT_DIR=/ceph/zoo/users/sminano/crabs_track_output

# version of the codebase
GIT_BRANCH=main

# Check if the target is not a directory
if [ ! -d "$VIDEO_DIR" ]; then
  exit 1
fi

# -----------------------------
# Create virtual environment
# -----------------------------
module load miniconda

# Define a environment for each job in the
# temporary directory of the compute node
ENV_NAME=crabs-dev-$SLURM_JOB_ID
ENV_PREFIX=$TMPDIR/$ENV_NAME

# create environment
conda create \
    --prefix $ENV_PREFIX \
    -y \
    python=3.10

# activate environment
conda activate $ENV_PREFIX

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

# ------------------------------------
# GPU specs
# ------------------------------------
echo "Memory used per GPU before training"
echo $(nvidia-smi --query-gpu=name,memory.total,memory.free,memory.used --format=csv) #noheader
echo "-----"

# -------------------
# Run evaluation script for each .mov file in VIDEO_DIR
# -------------------

TIMESTAMP=$(date +"%Y%m%d_%H%M%S")
PARENT_OUTPUT_DIR="${OUTPUT_DIR}_${TIMESTAMP}"
mkdir -p "$PARENT_OUTPUT_DIR"

for VIDEO_PATH in "$VIDEO_DIR"/*.mov; do
  VIDEO_BASENAME=$(basename "$VIDEO_PATH" .mov)

  echo "Processing video: $VIDEO_PATH"

  VIDEO_OUTPUT_DIR="$PARENT_OUTPUT_DIR/$VIDEO_BASENAME"

  mkdir -p "$VIDEO_OUTPUT_DIR"

  detect-and-track-video  \
    --trained_model_path "$TRAINED_MODEL_PATH" \
    --video_path "$VIDEO_PATH" \
    --config_file "$CONFIG_FILE" \
    --output_dir "$VIDEO_OUTPUT_DIR"

done
