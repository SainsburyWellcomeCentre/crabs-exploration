#!/bin/bash

#SBATCH -p gpu # a100 # partition
#SBATCH --gres=gpu:1 # gpu:a100_2g.10gb  # For any GPU: --gres=gpu:1. For a specific one: --gres=gpu:rtx5000
#SBATCH -N 1   # number of nodes
#SBATCH --ntasks-per-node 8 # 2 # max number of tasks per node
#SBATCH --mem 32G # memory pool for all cores
#SBATCH -t 3-00:00 # time (D-HH:MM)
#SBATCH -o slurm.%A.%N.out
#SBATCH -e slurm.%A.%N.err
#SBATCH --mail-type=ALL
#SBATCH --mail-user=s.minano@ucl.ac.uk


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

# mlflow
EXPERIMENT_NAME="Sept2023"
MLFLOW_FOLDER=/ceph/zoo/users/sminano/ml-runs-all/ml-runs-scratch

# dataset and train config
DATASET_DIR=/ceph/zoo/users/sminano/crabs_bboxes_labels/Sep2023_labelled
TRAIN_CONFIG_FILE=/ceph/zoo/users/sminano/cluster_train_config.yaml

# seeds for each dataset split
SPLIT_SEED=42

# version of the codebase
GIT_BRANCH=main


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

# ------------------------------------
# GPU specs
# ------------------------------------
echo "Memory used per GPU before training"
echo $(nvidia-smi --query-gpu=name,memory.total,memory.free,memory.used --format=csv) #noheader
echo "-----"

# -------------------
# Run training script
# -------------------
train-detector  \
 --dataset_dirs $DATASET_DIR \
 --config_file $TRAIN_CONFIG_FILE \
 --accelerator gpu \
 --experiment_name $EXPERIMENT_NAME \
 --seed_n $SPLIT_SEED \
 --mlflow_folder $MLFLOW_FOLDER \

# -----------------------------
# Delete virtual environment
# ----------------------------
conda deactivate
conda remove \
    --prefix $ENV_PREFIX \
    --all \
    -y
