#!/bin/bash

#SBATCH -p gpu # partition
#SBATCH -N 1   # number of nodes
#SBATCH --mem 64G # memory pool for all cores
#SBATCH -n 2 # number of cores
#SBATCH -t 3-00:00 # time (D-HH:MM)
#SBATCH --gres gpu:1 # request 1 GPU (of any kind)
#SBATCH -o slurm_array.%N.%A-%a.out
#SBATCH -e slurm_array.%N.%A-%a.err
#SBATCH --mail-type=ALL
#SBATCH --mail-user=s.minano@ucl.ac.uk
#SBATCH --array=0-6%4

#-------
# NOTE
# with "SBATCH --array=0-1%4" ---> runs n+1 separate jobs, but not more than m at a time.
# the number of array jobs should match the number of input files
#-------

# ---------------------
# Load required modules
# ----------------------
module load SLEAP

# ----------------------
# Input data
# ----------------------
# INPUT_DIR=/ceph/zoo/raw/CrabField/ramalhete_2021
# TODO: have list here?
# INPUT_DATA_LIST=($(<input.list))
INPUT_DATA_LIST=(
    "/ceph/zoo/raw/CrabField/ramalhete_2023/09.08.2023-Day2/09.08.2023-01-Left.MOV"
    "/ceph/zoo/raw/CrabField/ramalhete_2023/09.08.2023-Day2/09.08.2023-01-Right.mov"
    "/ceph/zoo/raw/CrabField/ramalhete_2023/09.08.2023-Day2/09.08.2023-02-Left.mov"
    "/ceph/zoo/raw/CrabField/ramalhete_2023/09.08.2023-Day2/09.08.2023-02-Right.MOV"
    "/ceph/zoo/raw/CrabField/ramalhete_2023/09.08.2023-Day2/09.08.2023-03-Left.MOV"
    "/ceph/zoo/raw/CrabField/ramalhete_2023/09.08.2023-Day2/09.08.2023-04-Left.MOV"
    "/ceph/zoo/raw/CrabField/ramalhete_2023/09.08.2023-Day2/09.08.2023-04-Right.MOV"
)

# Check len(list of input data) matches max SLURM_ARRAY_TASK_COUNT
# if not, exit
if [[ $SLURM_ARRAY_TASK_COUNT -ne ${#INPUT_DATA_LIST[@]} ]]; then
    echo "The number of array tasks does not match the number of inputs"
    exit 1
fi

# ----------------------
# output data location
# ----------------------
OUTPUT_DIR=/ceph/zoo/users/sminano/crabs_bboxes_labels
OUTPUT_SUBDIR="20230816_ramalhete2023_day2_combined"

# ----------------------
# parameters
# ----------------------
PARAM_VIDEO_EXT=MOV
PARAM_INI_SAMPLES=500
PARAM_SCALE=0.5
PARAM_N_COMPONENTS=5
PARAM_N_CLUSTERS=5
PARAM_PER_CLUSTER=10

# ----------------------
# script location
# ----------------------
# assumes repo located at '/ceph/scratch/sminano'
SCRATCH_PERSONAL_DIR=/ceph/scratch/sminano
SCRIPT_DIR=$SCRATCH_PERSONAL_DIR/crabs-exploration/crabs/bboxes_labelling


# -------------------
# Run python script
# -------------------
for i in {1..${SLURM_ARRAY_TASK_COUNT}}
do
    SAMPLE=${INPUT_DATA_LIST[${SLURM_ARRAY_TASK_ID}]}

    python "$SCRIPT_DIR"/extract_frames_to_label_w_sleap.py \
    $SAMPLE \
    --output_path $OUTPUT_DIR \
    --output_subdir $OUTPUT_SUBDIR \
    --video_extensions $PARAM_VIDEO_EXT \
    --initial_samples $PARAM_INI_SAMPLES \
    --scale $PARAM_SCALE \
    --n_components $PARAM_N_COMPONENTS \
    --n_clusters $PARAM_N_CLUSTERS \
    --per_cluster $PARAM_PER_CLUSTER \
    --compute_features_per_video
done
