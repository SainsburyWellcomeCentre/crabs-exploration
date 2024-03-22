#!/bin/bash

#SBATCH -p gpu # partition
#SBATCH -N 1   # number of nodes
#SBATCH -n 2 # number of cores
#SBATCH --mem 8G # memory pool for all cores
#SBATCH --gres=gpu:1  # any GPU
#SBATCH -t 3-00:00 # time (D-HH:MM)
#SBATCH -o slurm_%A-%N.out
#SBATCH -e slurm_%A-%N.err
#SBATCH --mail-type=ALL
#SBATCH --mail-user=s.minanon@ucl.ac.uk


# NOTE on SBATCH command for array jobs
# with "SBATCH --array=0-n%m" ---> runs n separate jobs, but not more than m at a time.
# the number of array jobs should match the number of input files

# ---------------------
# Source bashrc
# ----------------------

# Otherwise which python points to the module

source ~/.bashrc

# ---------------------
# Define variables
# ----------------------

# script location
CRABS_REPO_LOCATION=/ceph/scratch/sminano/crabs-exploration
DATASET_DIR=/ceph/zoo/users/sminano/crabs_bboxes_labels/Sep2023_labelled
TRAIN_CONFIG_FILE=/ceph/scratch/sminano/faster_rcnn.yaml

SPLIT_SEED=42

ENV_NAME=crabs-dev-$SPLIT_SEED-$SLURM_JOB_ID

# -----------------------------
# Create virtual environment
# -----------------------------

module load miniconda

conda create -n $ENV_NAME -y python=3.10
conda activate $ENV_NAME

which python
which pip

cd $CRABS_REPO_LOCATION
python -m pip install .

# -------------------
# Run training script
# -------------------
python "$CRABS_REPO_LOCATION"/crabs/detection_tracking/train_model.py  \
 --dataset_dirs $DATASET_DIR \
 --config_file $TRAIN_CONFIG_FILE \
 --accelerator gpu \
 --experiment_name "Sept2023_base_data_augm" \
 --seed_n $SEED_SPLIT \
