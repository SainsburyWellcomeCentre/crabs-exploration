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

# Run this script as
#   sbatch --array=0-n%m run_frame_extraction_w_entry_points.sh --config=input.json
#
# The idea is that this script changes as little as possible!
# Instead the input.json is the only file modified, and its content is printed to the logs
#
# NOTE for the optional argument "-array=0-n%m":
# runs n separate jobs, but not more than m at a time.
# the number of array jobs should match the number of input files


# ---------------------
# Create conda env
# ----------------------
# conda env create
# git clone repo
# pip install package



# ----------------------
# Input config
# ----------------------
# Print full json file to logs
# https://www.baeldung.com/linux/jq-command-json#1-prettify-json

# Check json
# Some config fields are mandatory


# Define defaults for optional fields
# To use if not defined in config
LOG_DIR=$OUTPUT_DIR/$OUTPUT_SUBDIR/logs
REENCODED_VIDEOS_SUBDIR=$REENCODED_VIDEOS_DIR/$OUTPUT_SUBDIR
# flag_reencode_input_videos

# ----------------------
# Input data
# ----------------------
# Read input videos from json file
# https://jqlang.github.io/jq/
# INPUT_DATA_LIST=()

# Check len(list of input data) matches max SLURM_ARRAY_TASK_COUNT
# if not, exit
if [[ $SLURM_ARRAY_TASK_COUNT -ne ${#INPUT_DATA_LIST[@]} ]]; then
    echo "The number of array tasks does not match the number of inputs"
    exit 1
fi


# ----------------------
# Output locations
# ----------------------
# Read output dir and subdir from json
# OUTPUT_DIR=/ceph/zoo/users/sminano/crabs_bboxes_labels
# OUTPUT_SUBDIR="Sep2023_day4_reencoded"

# Create location of SLURM logs
mkdir -p $LOG_DIR  # create if it doesnt exist

# read reencoding flag from json
# flag_reencode_input_videos
# https://stackoverflow.com/a/28185962

# Define location of reencoded videos if required
if [ "$flag_reencode_input_videos" = true ] ; then
    # Read reencoded dir from json
    # REENCODED_VIDEOS_DIR=/ceph/zoo/users/sminano/crabs_reencoded_videos
    # REENCODED_VIDEOS_SUBDIR=$REENCODED_VIDEOS_DIR/$OUTPUT_SUBDIR
    mkdir -p $REENCODED_VIDEOS_SUBDIR # create if it doesnt exist
fi


# ------------------------
# Command line tool
# ------------------------
for i in {1..${SLURM_ARRAY_TASK_COUNT}}
do
    # Input video
    SAMPLE=${INPUT_DATA_LIST[${SLURM_ARRAY_TASK_ID}]}
    echo "Input video: $SAMPLE"
    echo "--------"

    # --------------------------
    # Reencode video - if required (CLI tool)
    # --------------------------
    echo "Reencoding ..."
    reencode-video ...

    # # Check status
    # if [ "$?" -ne 0 ]; then
    #     echo "Reencoding failed! Please check .err log"
    # else
    #     echo "Reencoded video: $REENCODED_VIDEO_PATH"
    # fi
    # echo "--------"


    # -------------------
    # Extract frames
    # -------------------
    echo Extracting frames
    extract-frames ...

    # # Check status
    # if [ "$?" -ne 0 ]; then
    #     echo "Frame extraction failed! Please check .err log"
    # else
    #     echo "Frames extracted from video: $FRAME_EXTRACTION_INPUT_VIDEO"
    # fi
    # echo "--------"


    # -------------------
    # Logs
    # -------------------
    # Reencoded videos log
    # copy .err file to go with reencoded video too if required
    # filename: {reencoded video name}.{slurm_array}.{slurm_job_id}
    # TODO: make a nicer log
    if [ "$flag_reencode_input_videos" = true ] ; then
        for ext in err out
        do
            cp slurm_array.$SLURMD_NODENAME.$SLURM_ARRAY_JOB_ID-$SLURM_ARRAY_TASK_ID.$ext \
            /$REENCODED_VIDEOS_SUBDIR/"$filename_no_ext"_RE.slurm_array.$SLURM_ARRAY_JOB_ID-$SLURM_ARRAY_TASK_ID.$ext
        done
    fi

    # Frame extraction logs
    # Move logs for this job to subdir with extracted frames
    for ext in err out
    do
        mv slurm_array.$SLURMD_NODENAME.$SLURM_ARRAY_JOB_ID-$SLURM_ARRAY_TASK_ID.$ext /$LOG_DIR
    done
done
