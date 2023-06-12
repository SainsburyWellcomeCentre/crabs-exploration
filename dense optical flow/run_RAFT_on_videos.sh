#!/bin/bash 

#SBATCH -p gpu # partition
#SBATCH -N 1   # number of nodes
#SBATCH --mem 12G # memory pool for all cores
#SBATCH -n 2 # number of cores
#SBATCH -t 0-23:00 # time (D-HH:MM)
#SBATCH --gres gpu:1 # request 1 GPU (of any kind)
#SBATCH -o slurm.%N.%j.out # write STDOUT
#SBATCH -e slurm.%N.%j.err # write STDERR
#SBATCH --mail-type=ALL
#SBATCH --mail-user=s.minano@ucl.ac.uk  

# ---------------------
# Load required modules
# ----------------------
module load cuda
module load miniconda

# -------------------
# Clone repo
# -------------------
SCRATCH_CRABS_DIR=/ceph/scratch/sminano/crabs_optical_flow
cd $SCRATCH_CRABS_DIR

git clone https://github.com/princeton-vl/RAFT.git
RAFT_REPO_ROOT_DIR=$SCRATCH_CRABS_DIR/RAFT


# --------------------------
# Set up conda environment
# ---------------------------
# pip install pathlib
conda create --name raft
conda activate raft
conda install pytorch torchvision cudatoolkit -c pytorch
conda install scipy

# the following with pip because I get an error with conda
# that requires to update base conda...which is common in
# the cluster?
pip install opencv-python
pip install matplotlib
pip install tensorboard
pip install pathlib

# -------------------
# Run python script
# -------------------
# Input data
# NINJAV_S001_S001_T003_subclip.mp4
INPUT_DATA_DIR=$SCRATCH_CRABS_DIR/data/ 
# output dir
OUTPUT_DIR=$SCRATCH_CRABS_DIR/output/

# Download models
cd $RAFT_REPO_ROOT_DIR 
./download_models.sh
MODEL_PATH=$RAFT_REPO_ROOT_DIR/models/raft-kitti.pth

# copy data and models from scratch to temp? (faster)

# run python script
cd ..
STEP_FRAMES=10
python estimate_optical_flow_on_video.py \
 --model $MODEL_PATH \
 --input_dir $INPUT_DATA_DIR \
 --output_dir $OUTPUT_DIR \
 --step_frames $STEP_FRAMES

