#!/bin/bash

#SBATCH -p gpu # partition
#SBATCH -N 1   # number of nodes
#SBATCH --mem 32G # memory pool for all cores
#SBATCH -n 2 # number of cores
#SBATCH -t 3-04:00 # time (D-HH:MM)
#SBATCH --gres gpu:1 # request 1 GPU (of any kind)
#SBATCH -o slurm.%N.%j.out # write STDOUT
#SBATCH -e slurm.%N.%j.err # write STDERR
#SBATCH --mail-type=ALL
#SBATCH --mail-user=s.minano@ucl.ac.uk

# Load the SLEAP module
module load SLEAP

# training package directory
DATA_DIR=/ceph/zoo/users/sminano/crabs_pose_4k_TD4
JOB_DIR=$DATA_DIR/labels.v001.slp.training_job

# inference video location ("In Domain" - IDom)
INFER_DIR_NAME=Camera2
INFER_VIDEO_NAME=NINJAV_S001_S001_T010.MOV
INFER_VIDEO_PATH=/ceph/zoo/raw/CrabField/swc-courtyard_2023/$INFER_DIR_NAME/$INFER_VIDEO_NAME

# Go to the training package directory
cd $JOB_DIR

# Run the inference command
sleap-track $INFER_VIDEO_PATH \
    -m $JOB_DIR/models/230725_174219.centroid/training_config.json \
    -m $JOB_DIR/models/230725_174219.centered_instance/training_config.json \
    -o $INFER_DIR_NAME-$INFER_VIDEO_NAME.predictions.slp \
    --frames 2000-6000 \
    --verbosity json \
    --no-empty-frames \
    --tracking.tracker none \
    --gpu auto \
    --max_instances 1 \
    --batch_size 4
