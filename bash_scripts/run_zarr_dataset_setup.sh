#!/bin/bash

#SBATCH -p cpu
#SBATCH -N 1
#SBATCH --ntasks-per-node 1
#SBATCH --mem 8G
#SBATCH -t 0-01:00
#SBATCH -o slurm_setup.%j.%N.out
#SBATCH -e slurm_setup.%j.%N.err
#SBATCH --mail-type=ALL
#SBATCH --mail-user=s.minano@ucl.ac.uk

set -e
set -u
set -o pipefail

# ---------------------
# Define common variables here
# ----------------------

VIA_TRACKS_DIR="/ceph/zoo/users/sminano/loops_tracking_above_10th_percentile_slurm_1825237_2071125_2071084"
METADATA_CSV="/ceph/zoo/processed/CrabField/ramalhete_2023/CrabLabels/crab-loops/loop-frames-ffmpeg.csv"

# ZARR_STORE_OUTPUT -- defined per subjob
ZARR_MODE_STORE="a"
ZARR_MODE_GROUP="w-"

GIT_BRANCH=smg/reset-individual-numbers-bef-merge

# -----------------------------
# Create virtual environment
# -----------------------------
# Common virtual environment to all array jobs
module load miniconda

ENV_PREFIX=/ceph/zoo/users/sminano/envs/crabs-zarr-$SLURM_JOB_ID

CONDA_PKGS_DIRS=/tmp/conda-pkgs-$SLURM_JOB_ID conda create \
    --prefix $ENV_PREFIX \
    -y \
    python=3.12

source activate $ENV_PREFIX

python -m pip install git+https://github.com/SainsburyWellcomeCentre/crabs-exploration.git@$GIT_BRANCH

echo "Git branch: $GIT_BRANCH"
conda list crabs
echo "Environment created at $ENV_PREFIX"

# --------------------------------------------
# Submit array job, passing all variables
# -------------------------------------------
ARRAY_JOB_ID=$(
    sbatch --parsable sbatch \
    --dependency=afterok:$SLURM_JOB_ID \
    --export=ALL,VIA_TRACKS_DIR=$VIA_TRACKS_DIR,METADATA_CSV=$METADATA_CSV,ZARR_MODE_STORE=$ZARR_MODE_STORE,ZARR_MODE_GROUP=$ZARR_MODE_GROUP,GIT_BRANCH=$GIT_BRANCH,ENV_PREFIX=$ENV_PREFIX \
    run_zarr_dataset_creation.sh
)

echo "Setup job ID:  $SLURM_JOB_ID"
echo "Array job ID:  $ARRAY_JOB_ID"
