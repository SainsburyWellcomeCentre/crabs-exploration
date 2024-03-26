#!/bin/bash

#SBATCH -p gpu # partition
#SBATCH -N 1   # number of nodes
#SBATCH --ntasks-per-node 2 # number of tasks per node
#SBATCH --mem 64G # memory pool for all cores
#SBATCH --gres=gpu:1  # any GPU
#SBATCH -t 3-00:00 # time (D-HH:MM)
#SBATCH -o slurm.%A.%N.out
#SBATCH -e slurm.%A.%N.err
#SBATCH --mail-type=ALL
#SBATCH --mail-user=s.minanon@ucl.ac.uk


# ---------------------
# Source bashrc
# ----------------------
# Otherwise `which python` points to the miniconda module's Python
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

# dataset and train config
DATASET_DIR=/ceph/zoo/users/sminano/crabs_bboxes_labels/Sep2023_labelled
TRAIN_CONFIG_FILE=/ceph/scratch/sminano/crabs-exploration/cluster_train_config.yaml

# seed for the dataset split
SPLIT_SEED=42

# version of the codebase
GIT_BRANCH=smg/train-entry-point

# -----------------------------
# Create virtual environment
# -----------------------------

module load miniconda

# create a unique environment for this job
ENV_NAME=crabs-dev-$SPLIT_SEED-$SLURM_JOB_ID
conda create -n $ENV_NAME -y python=3.10
conda activate $ENV_NAME

# check pip and python
which python
which pip

# install crabs package in virtual env
# $HOME/.conda/envs/$ENV_NAME
# cd $CRABS_REPO_LOCATION
python -m pip install git+https://github.com/SainsburyWellcomeCentre/crabs-exploration.git@$GIT_BRANCH

# -------------------
# Run training script
# -------------------
train-detector  \
 --dataset_dirs $DATASET_DIR \
 --config_file $TRAIN_CONFIG_FILE \
 --accelerator gpu \
 --experiment_name "Sept2023_base_data_augm" \
 --seed_n $SPLIT_SEED \

# -----------------------------
# Delete virtual environment
# -----------------------------
conda deactivate
conda remove -n $ENV_NAME --all
