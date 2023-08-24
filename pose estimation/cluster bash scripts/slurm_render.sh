#!/bin/bash

#SBATCH -p gpu # partition
#SBATCH -N 1   # number of nodes
#SBATCH --mem 12G # memory pool for all cores
#SBATCH -n 2 # number of cores
#SBATCH -t 3-04:00 # time (D-HH:MM)
#SBATCH --gres gpu:1 # request 1 GPU (of any kind)
#SBATCH -o slurm.%N.%j.out # write STDOUT
#SBATCH -e slurm.%N.%j.err # write STDERR
#SBATCH --mail-type=ALL
#SBATCH --mail-user=s.minano@ucl.ac.uk

# Load the SLEAP module
module load SLEAP

# inference video location ("In Domain" - IDom)
DATA_DIR=/ceph/zoo/users/sminano/crabs_pose_4k_TD4
JOB_DIR=$DATA_DIR/labels.v001.slp.training_job
PREDICTIONS_PATH=$JOB_DIR/Camera2-NINJAV_S001_S001_T010.MOV.predictions.slp


# render video
sleap-render $PREDICTIONS_PATH --frames 2000-6000 \
    --distinctly_color nodes \
    --marker_size 1 \
    --show_edges 0 \
    --fps 60
