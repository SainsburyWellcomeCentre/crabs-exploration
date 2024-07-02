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

# -----------------------------
# Error settings for bash
# -----------------------------
# see https://wizardzines.com/comics/bash-errors/
set -e  # do not continue after errors
set -u  # throw error if variable is unset
set -o pipefail  # make the pipe fail if any part of it fails

# ---------------------
# Source bashrc
# ----------------------
# Otherwise `which python` points to the miniconda module's Python
# source ~/.bashrc #


# ---------------------
# Define variables
# ----------------------

# mlflow
EXPERIMENT_NAME="Sept2023_data_augm"
MLFLOW_FOLDER=/ceph/zoo/users/sminano/ml-runs-all/ml-runs-scratch

# dataset and configs directories
DATASET_DIR=/ceph/zoo/users/sminano/crabs_bboxes_labels/Sep2023_labelled
CONFIGS_DIR=/ceph/zoo/users/sminano/crabs_data_augmentation_configs

# version of the codebase
GIT_BRANCH=main


# -----------------------------------------
# Parameter sweep
# -----------------------------------------
# from this great gist:
# https://gist.github.com/TysonRayJones/1c4cae5acd7fde3a90da743cbb79db2e
list_seeds=($(echo {42..44}))
list_config_files=("$CONFIGS_DIR"/*.yaml)

len_seeds=${#list_seeds[@]}
len_config_files=${#list_config_files[@]}
n_jobs=$((len_seeds * len_config_files))

# Check len(list of input data) matches max SLURM_ARRAY_TASK_COUNT
# if not, exit
if [[ $SLURM_ARRAY_TASK_COUNT -ne $n_jobs ]]; then
    echo "The number of array tasks ($SLURM_ARRAY_TASK_COUNT) does not match "
    echo "the number of parameter combinations to sweep across ($n_jobs)."
    exit 1
fi

# Get params for this job
trial_dummy=${SLURM_ARRAY_TASK_ID}  # initialise variable
config=${list_config_files[$(( trial_dummy % ${#list_config_files[@]} ))]}
trial_dummy=$(( trial_dummy / ${#list_config_files[@]} ))
seed=${list_seeds[$(( trial_dummy % ${#list_seeds[@]} ))]}

echo "-----------------"
echo "Inputs for $SLURM_ARRAY_JOB_ID-$SLURM_ARRAY_TASK_ID "
echo "trial: ${SLURM_ARRAY_TASK_ID}"
echo "config: $config"
echo "seed: $seed"
echo "-----------------"


# -----------------------------
# Create virtual environment
# -----------------------------
export PYTHONNOUSERSITE=True
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
# replace conda --> source, otherwise conda activate not found?
source activate $ENV_PREFIX

# log pip and python locations
echo $ENV_PREFIX
which python  # should be python of the environment
which pip  # should be pip of the environment
echo "-----------------"

# install crabs package in virtual env
# pip install --upgrade pip ---> python -m pip install --upgrade pip?
python -m pip install git+https://github.com/SainsburyWellcomeCentre/crabs-exploration.git@$GIT_BRANCH

# print the version of crabs package (last number is the commit hash)
echo "Git branch: $GIT_BRANCH"
conda list crabs
echo "-----------------"

# ------------------------------------
# GPU specs
# ------------------------------------
echo "Memory used per GPU before training"
echo $(nvidia-smi --query-gpu=name,memory.total,memory.free,memory.used --format=csv) #noheader
echo "-----------------"

# -------------------
# Run training script
# -------------------
train-detector  \
 --dataset_dirs $DATASET_DIR \
 --config_file $config \
 --accelerator gpu \
 --experiment_name $EXPERIMENT_NAME \
 --seed_n $seed \
 --mlflow_folder $MLFLOW_FOLDER \
 --log_data_augmentation
