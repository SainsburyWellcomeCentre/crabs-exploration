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
 #SBATCH --mail-user=n.aznan@ucl.ac.uk

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
 VIDEO_PATH=/ceph/zoo/users/sminano/crabs_tracks_label/04.09.2023-04-Right_RE_test/04.09.2023-04-Right_RE_test_video.mp4
 CONFIG_FILE=/ceph/scratch/nikkna/crabs-exploration/crabs/tracker/config/tracking_config.yaml

 # checkpoint
 TRAINED_MODEL_PATH=/ceph/scratch/nikkna/crabs-exploration/ml_ckpt/595664011639950974/e24234398e4b4d5790a9ea3599570637/checkpoints/last.ckpt

 # output directory
 OUTPUT_DIR=/ceph/scratch/nikkna/crabs-exploration/crabs_track_output

 # ground truth is available
 GT_PATH=/ceph/zoo/users/sminano/crabs_tracks_label/04.09.2023-04-Right_RE_test/04.09.2023-04-Right_RE_test_corrected_ST_csv.csv

 # version of the codebase
 GIT_BRANCH=nikkna/inference_cluster

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
 # Run evaluation script
 # -------------------
 detect-and-track-video  \
  --trained_model_path $TRAINED_MODEL_PATH \
  --video_path $VIDEO_PATH \
  --config_file $CONFIG_FILE \
  --output_dir $OUTPUT_DIR \
  --gt_path $GT_PATH
