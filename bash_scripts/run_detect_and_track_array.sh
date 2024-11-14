#!/bin/bash

#SBATCH -p gpu # # partition
#SBATCH --gres=gpu:1 # For any GPU: --gres=gpu:1. For a specific one: --gres=gpu:rtx5000
#SBATCH -N 1   # number of nodes
#SBATCH --ntasks-per-node 8 # 2 # max number of tasks per node
#SBATCH --mem 32G # memory pool for all cores
#SBATCH -t 3-00:00 # time (D-HH:MM)
#SBATCH -o slurm_array.%A-%a.%N.out
#SBATCH -e slurm_array.%A-%a.%N.err
#SBATCH --mail-type=ALL
#SBATCH --mail-user=s.minano@ucl.ac.uk
#SBATCH --array=0-4%3


# NOTE on SBATCH command for array jobs
# with "SBATCH --array=0-n%m" ---> runs n separate jobs, but not more than m at a time.
# the number of array jobs should match the number of input files

# ---------------------
# Source bashrc
# ----------------------
# Otherwise `which python` points to the miniconda module's Python
# source ~/.bashrc


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

# Path to the trained model
CKPT_PATH="/ceph/zoo/users/sminano/ml-runs-all/ml-runs/317777717624044570/40b1688a76d94bd08175cb380d0a6e0e/checkpoints/last.ckpt"

# Path to the tracking config file
TRACKING_CONFIG_FILE="/ceph/zoo/users/sminano/cluster_tracking_config.yaml"

# Path to the groundtruth annotations file
GROUNDTRUTH_ANNOTATIONS_FILE="/ceph/zoo/users/sminano/tracking_groundtruth/04.09.2023-04-Right_RE_test_corrected_ST_csv_SM_corrected_20241029_113207.csv"

# List of videos to run inference on: define VIDEOS_DIR and VIDEO_FILENAME
# NOTE: if any of the paths have spaces, put the path in quotes, but stopping and re-starting at the wildcard.
# e.g.: "/ceph/zoo/users/sminano/ml-runs-all/ml-runs-scratch/763954951706829194/"*"/checkpoints"
# e.g.: "checkpoint-epoch="*".ckpt"
# List of videos to run inference on
VIDEOS_DIR="/ceph/zoo/users/sminano/escape_clips_sample"
VIDEO_FILENAME=*.mov
mapfile -t LIST_VIDEOS < <(find $VIDEOS_DIR -type f -name $VIDEO_FILENAME)

# Select output
SAVE_FRAMES=true
SAVE_VIDEO=true

# version of the codebase
GIT_BRANCH=main

# --------------------
# Check inputs
# --------------------
# Check len(list of input data) matches max SLURM_ARRAY_TASK_COUNT
# if not, exit
if [[ $SLURM_ARRAY_TASK_COUNT -ne ${#LIST_VIDEOS[@]} ]]; then
    echo "The number of array tasks does not match the number of input videos"
    exit 1
fi

# -----------------------------
# Create virtual environment
# -----------------------------
module load miniconda

# Define a environment for each job in the
# temporary directory of the compute node
ENV_NAME=crabs-dev-$SLURM_ARRAY_JOB_ID-$SLURM_ARRAY_TASK_ID
ENV_PREFIX=$TMPDIR/$ENV_NAME

# create environment
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

# ------------------------------------
# GPU specs
# ------------------------------------
echo "Memory used per GPU before training"
echo $(nvidia-smi --query-gpu=name,memory.total,memory.free,memory.used --format=csv) #noheader
echo "-----"


# -------------------------
# Run evaluation script
# -------------------------
# video used in this job
INPUT_VIDEO=${LIST_VIDEOS[${SLURM_ARRAY_TASK_ID}]}

echo "Running inference on $INPUT_VIDEO using trained model at $CKPT_PATH"

# append relevant flag to command if the test set is to be used
# if [ "$SAVE_FRAMES" = "true" ]; then
#     SAVE_FRAMES_FLAG="--save_frames"
# else
#     SAVE_FRAMES_FLAG=""
# fi

# if [ "$SAVE_VIDEO" = "true" ]; then
#     SAVE_VIDEO_FLAG="--save_video"
# else
#     SAVE_VIDEO_FLAG=""
# fi
# Set flags based on boolean variables
SAVE_FRAMES_FLAG=${SAVE_FRAMES:+--save_frames}
SAVE_VIDEO_FLAG=${SAVE_VIDEO:+--save_video}

# run detect-and-track command
detect-and-track-video  \
    --trained_model_path $CKPT_PATH  \
    --video_path $INPUT_VIDEO  \
    --config_file $TRACKING_CONFIG_FILE  \
    --annotations_file $GROUNDTRUTH_ANNOTATIONS_FILE  \
    --accelerator gpu  \
    $SAVE_FRAMES_FLAG  \
    $SAVE_VIDEO_FLAG



# copy tracking config to output directory
shopt -s extglob  # Enable extended globbing
TARGET_DIR=$(echo tracking_output_* | awk '{print $1}')

cp "$TRACKING_CONFIG_FILE" "$TARGET_DIR"/"$TRACKING_CONFIG_FILE"_$SLURM_ARRAY_JOB_ID-$SLURM_ARRAY_TASK_ID
echo "Copied $TRACKING_CONFIG_FILE to $TARGET_DIR"
