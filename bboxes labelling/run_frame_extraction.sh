#!/bin/bash 

#SBATCH -p gpu # partition
#SBATCH -N 1   # number of nodes
#SBATCH --mem 12G # memory pool for all cores
#SBATCH -n 2 # number of cores
#SBATCH -t 3-00:00 # time (D-HH:MM)
#SBATCH --gres gpu:1 # request 1 GPU (of any kind)
#SBATCH -o slurm.%N.%j.out # write STDOUT
#SBATCH -e slurm.%N.%j.err # write STDERR
#SBATCH --mail-type=ALL
#SBATCH --mail-user=s.minano@ucl.ac.uk  

# ---------------------
# Load required modules
# ----------------------
# Option 1
# module load cuda/11.8
# module load miniconda
# conda activate extract-frames  # ----> environment as defined in sleap installation

# Option 2:
module load SLEAP

# ---------------------
# Define environment variables
# ----------------------
SCRATCH_PERSONAL_DIR=/ceph/scratch/sminano
INPUT_DIR=$SCRATCH_PERSONAL_DIR/crabs_sample/videos_inference  #--------- CHANGE
OUTPUT_DIR=$SCRATCH_PERSONAL_DIR/crabs_bbox_labels

# --------------------------
# Set up conda environment
# ---------------------------
# conda activate extract-frames
# TODO: re-creates it everytime?
# use create -p instead?
# conda create --name extract-frames python=3.7
# conda activate extract-frames
# conda install -c sleap -c nvidia -c conda-forge sleap
# conda install -c conda-forge opencv

# TODO: set NUMEXPR_MAX_THREADS?
# NumExpr detected 40 cores but "NUMEXPR_MAX_THREADS" not set, 
# so enforcing safe limit of 8.

# -------------------
# Run python script
# -------------------
cd $OUTPUT_DIR
python extract_frames_to_label_w_sleap.py \
 $INPUT_DIR \
 --output_path $OUTPUT_DIR \
 --video_extensions 'MOV' \
 --initial_samples 300 \
 --n_components 5 \
 --n_clusters 5 \
 --per_cluster 5 \
 --compute_features_per_video