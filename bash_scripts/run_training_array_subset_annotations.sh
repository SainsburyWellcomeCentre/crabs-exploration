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
#SBATCH --array=0-14%5


# NOTE on SBATCH command for array jobs
# with "SBATCH --array=0-n%m" ---> runs n separate jobs, but not more than m at a time.
# the number of array jobs should match the number of input files

# To exclude specific nodes from being used, use the --exclude flag:
#SBATCH --exclude=gpu-380-12,gpu-350-01

# ---------------------
# Source bashrc
# ----------------------
# Otherwise `which python` points to the miniconda module's Python
# needs to be before error setting?
source ~/.bashrc


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
EXPERIMENT_NAME="Sept2023_remove_small_bboxes"
MLFLOW_FOLDER=/ceph/zoo/users/sminano/ml-runs-all/ml-runs-scratch

# dataset and train config
DATASET_DIR=/ceph/zoo/users/sminano/crabs_bboxes_labels/Sep2023_labelled
ANNOTATIONS_DIR=/ceph/zoo/users/sminano/crabs_subset_annotations/large_annotations
TRAIN_CONFIG_FILE=/ceph/zoo/users/sminano/crabs_data_augmentation_configs/01_config_all_data_augmentation.yaml

# version of the codebase
GIT_BRANCH=main

# -----------------------------------------
# Parameter sweep
# -----------------------------------------
# from this great gist:
# https://gist.github.com/TysonRayJones/1c4cae5acd7fde3a90da743cbb79db2e
list_seeds=($(echo {42..44}))
list_annotation_files=("$ANNOTATIONS_DIR"/*.json)

len_seeds=${#list_seeds[@]}
len_annotation_files=${#list_annotation_files[@]}
n_jobs=$((len_seeds * len_annotation_files))

# Check len(list of input data) matches max SLURM_ARRAY_TASK_COUNT
# if not, exit
if [[ $SLURM_ARRAY_TASK_COUNT -ne $n_jobs ]]; then
    echo "The number of array tasks ($SLURM_ARRAY_TASK_COUNT) does not match "
    echo "the number of annotation files times seeds ($n_jobs)."
    exit 1
fi

# Get params for this job
# - seed is the inner loop
# - annotation_file is the outer loop
trial_dummy=${SLURM_ARRAY_TASK_ID}  # initialise variable
seed=${list_seeds[$(( trial_dummy % ${#list_seeds[@]} ))]}

trial_dummy=$(( trial_dummy / ${#list_seeds[@]} ))
annotation_file=${list_annotation_files[$(( trial_dummy % ${#list_annotation_files[@]} ))]}
# trial=$(( trial / ${#list_annotation_files[@]} ))

echo "-----------------"
echo "Inputs for $SLURM_ARRAY_JOB_ID-$SLURM_ARRAY_TASK_ID "
echo "trial: ${SLURM_ARRAY_TASK_ID}"
echo "annotation_file: $annotation_file"
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
    python=3.12

# activate environment
source activate $ENV_PREFIX


# log pip and python locations
echo $ENV_PREFIX
which python
which pip
echo "-----------------"

# install crabs package in virtual env
python -m pip install git+https://github.com/SainsburyWellcomeCentre/crabs-exploration.git@$GIT_BRANCH


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
 --annotation_files $annotation_file \
 --config_file $TRAIN_CONFIG_FILE \
 --accelerator gpu \
 --experiment_name $EXPERIMENT_NAME \
 --seed_n $seed \
 --mlflow_folder $MLFLOW_FOLDER \
 --log_data_augmentation

# -----------------------------
# Delete virtual environment
# ----------------------------
conda deactivate
conda remove \
    --prefix $ENV_PREFIX \
    --all \
    -y
