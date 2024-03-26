#!/bin/bash

#SBATCH -p gpu # partition
#SBATCH -N 1   # number of nodes
#SBATCH --mem 8G # memory pool for all cores
#SBATCH -n 2 # number of cores
#SBATCH -t 3-00:00 # time (D-HH:MM)
#SBATCH -o slurm_array.%N.%A-%a.out
#SBATCH -e slurm_array.%N.%A-%a.err
#SBATCH --mail-type=ALL
#SBATCH --mail-user=s.minanon@ucl.ac.uk
#SBATCH --array=0-2%3


# NOTE on SBATCH command for array jobs
# with "SBATCH --array=0-n%m" ---> runs n separate jobs, but not more than m at a time.
# the number of array jobs should match the number of input files

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

# Input files
CRABS_REPO_LOCATION=/ceph/scratch/sminano/crabs-exploration
DATASET_DIR=/ceph/zoo/users/sminano/crabs_bboxes_labels/Sep2023_labelled
TRAIN_CONFIG_FILE=/ceph/scratch/sminano/faster_rcnn.yaml

LIST_SEEDS=($(echo {42..44}))
SEED_SPLIT=${LIST_SEEDS[${SLURM_ARRAY_TASK_ID}]}

ENV_NAME=crabs-dev-$SEED_SPLIT

# --------------------
# Check inputs
# --------------------
# Check len(list of input data) matches max SLURM_ARRAY_TASK_COUNT
# if not, exit
if [[ $SLURM_ARRAY_TASK_COUNT -ne ${#LIST_SEEDS[@]} ]]; then
    echo "The number of array tasks does not match the number of inputs"
    exit 1
fi

# -----------------------------
# Create virtual environment
# -----------------------------
module load miniconda

conda create -y -n $ENV_NAME python=3.10
source activate $ENV_NAME

cd $CRABS_REPO_LOCATION
python -m pip install .[dev]  # can omit -e if not debugging


# -------------------
# Run training script
# -------------------
python "$CRABS_REPO_LOCATION"/crabs/detection_tracking/train_model.py  \
 --dataset_dirs $DATASET_DIR \
 --config_file $TRAIN_CONFIG_FILE \
 --accelerator gpu \
 --experiment_name "Sept2023_base_data_augm" \
 --seed_n $SEED_SPLIT \