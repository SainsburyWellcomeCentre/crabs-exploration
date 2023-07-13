#!/bin/bash 

#SBATCH -p gpu # partition
#SBATCH -N 1   # number of nodes
#SBATCH --mem 32G # memory pool for all cores
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
module load SLEAP

# ---------------------
# Define environment variables
# ----------------------
# input/output dirs
INPUT_DIR=/ceph/zoo/raw/CrabField/ramalhete_2021
OUTPUT_DIR=/ceph/zoo/users/sminano/crabs_bboxes_labels

# script location
# assumes repo located at '/ceph/scratch/sminano'
SCRATCH_PERSONAL_DIR=/ceph/scratch/sminano
SCRIPT_DIR=$SCRATCH_PERSONAL_DIR/crabs-exploration/"bboxes labelling"

# TODO: set NUMEXPR_MAX_THREADS?
# NumExpr detected 40 cores but "NUMEXPR_MAX_THREADS" not set, 
# so enforcing safe limit of 8.

# -------------------
# Run python script
# -------------------
python "$SCRIPT_DIR"/extract_frames_to_label_w_sleap.py \
 x/camera1 $INPUT_DIR/camera2/NINJAV_S001_S001_T001.MOV  $INPUT_DIR/camera2/NINJAV_S001_S001_T002.MOV \
 --output_path $OUTPUT_DIR \
 --video_extensions MOV \
 --initial_samples 200 \
 --n_components 5 \
 --n_clusters 5 \
 --per_cluster 8 \
 --compute_features_per_video