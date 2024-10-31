#!/bin/bash

#SBATCH -p gpu # a100 # partition
#SBATCH --gres=gpu:1 # gpu:a100_2g.10gb  # For any GPU: --gres=gpu:1. For a specific one: --gres=gpu:rtx5000
#SBATCH -N 1   # number of nodes
#SBATCH --ntasks-per-node 8 # 2 # max number of tasks per node
#SBATCH --mem 32G # memory pool for all cores
#SBATCH -t 3-00:00 # time (D-HH:MM)
#SBATCH -o slurm_array.%A-%a.%N.out
#SBATCH -e slurm_array.%A-%a.%N.err
#SBATCH --mail-type=ALL
#SBATCH --mail-user=s.minano@ucl.ac.uk
#SBATCH --array=0-2%3


# NOTE on SBATCH command for array jobs
# with "SBATCH --array=0-n%m" ---> runs n separate jobs, but not more than m at a time.
# the number of array jobs should match the number of input files

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

# List of models to evaluate
MLFLOW_CKPTS_FOLDER=/ceph/zoo/users/sminano/ml-runs-all/ml-runs/317777717624044570/fe9a6c2f491a4496aade5034c75316cc/checkpoints
LIST_CKPT_FILES=("$MLFLOW_CKPTS_FOLDER"/*.ckpt)

# selected model
CKPT_PATH=${LIST_CKPT_FILES[${SLURM_ARRAY_TASK_ID}]}

# destination mlflow folder
# EXPERIMENT_NAME="Sept2023" ----> get from training job
MLFLOW_FOLDER=/ceph/zoo/users/sminano/ml-runs-all/ml-runs

# version of the codebase
GIT_BRANCH=main

# --------------------
# Check inputs
# --------------------
# Check len(list of input data) matches max SLURM_ARRAY_TASK_COUNT
# if not, exit
if [[ $SLURM_ARRAY_TASK_COUNT -ne ${#LIST_CKPT_FILES[@]} ]]; then
    echo "The number of array tasks does not match the number of .ckpt files"
    exit 1
fi

# -----------------------------
# Create virtual environment
# -----------------------------
module load miniconda

# Define a environment for each job in the
# temporary directory of the compute node
ENV_NAME=crabs-dev-$SPLIT_SEED-$SLURM_ARRAY_JOB_ID
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
# Run training script
# -------------------
echo "Evaluating trained model at $CKPT_PATH: "
evaluate-detector  \
 --trained_model_path $CKPT_PATH \
 --accelerator gpu \
 --mlflow_folder $MLFLOW_FOLDER \
echo "-----"