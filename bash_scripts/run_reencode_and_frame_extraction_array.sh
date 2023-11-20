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
#SBATCH --array=0-0%5

#-------
# NOTE!!
# with "SBATCH --array=0-n%m" ---> runs n separate jobs, but not more than m at a time.
# the number of array jobs should match the number of input files

# ---------------------
# Load required modules
# ----------------------
module load SLEAP

# ----------------------
# Input data
# ----------------------
# INPUT_DIR=/ceph/zoo/raw/CrabField/ramalhete_2023
# # TODO: have list here? change to directory?
# INPUT_DATA_LIST=($(<input.list))
INPUT_DATA_LIST=(
    "/ceph/zoo/raw/CrabField/ramalhete_2023/04.09.2023-Day1/04.09.2023-05-Left.mp4"
)
# Check len(list of input data) matches max SLURM_ARRAY_TASK_COUNT
# if not, exit
if [[ $SLURM_ARRAY_TASK_COUNT -ne ${#INPUT_DATA_LIST[@]} ]]; then
    echo "The number of array tasks does not match the number of inputs"
    exit 1
fi

# set whether to reencode input videos or not
flag_reencode_input_videos=false
reencoded_extension=mp4

# ----------------------
# Output data location
# ----------------------
# location of extracted frames
# TODO: derive subdir name from parent dir
OUTPUT_DIR=/ceph/zoo/users/sminano/crabs_bboxes_labels
OUTPUT_SUBDIR="Sep2023_day1_05_Left_reencoded"

# location of SLURM logs
LOG_DIR=$OUTPUT_DIR/$OUTPUT_SUBDIR/logs
mkdir -p $LOG_DIR  # create if it doesnt exist

# set location of reencoded videos if required
if [ "$flag_reencode_input_videos"=true ] ; then
    REENCODED_VIDEOS_DIR=/ceph/zoo/users/sminano/crabs_reencoded_videos
    REENCODED_VIDEOS_SUBDIR=$REENCODED_VIDEOS_DIR/$OUTPUT_SUBDIR 
    mkdir -p $REENCODED_VIDEOS_SUBDIR # create if it doesnt exist
fi
# ---------------------------------
# Frame extraction parameters
# -----------------------------------
# extension of the videos from which frames are extracted! 
if [ "$flag_reencode_input_videos"=true ] ; then
    PARAM_VIDEO_EXT=$reencoded_extension 
else
    PARAM_VIDEO_EXT=MOV # TODO: derive video extension if not provided?
fi
PARAM_INI_SAMPLES=500
PARAM_SCALE=0.5
PARAM_N_COMPONENTS=5
PARAM_N_CLUSTERS=5
PARAM_PER_CLUSTER=4


# ----------------------
# Script location
# ----------------------
# assumes repo located at '/ceph/scratch/sminano'
SCRATCH_PERSONAL_DIR=/ceph/scratch/sminano
SCRIPT_DIR=$SCRATCH_PERSONAL_DIR/crabs-exploration/crabs/bboxes_labelling

# -------------------
# Run python script
# -------------------
for i in {1..${SLURM_ARRAY_TASK_COUNT}}
do
    # Input video
    SAMPLE=${INPUT_DATA_LIST[${SLURM_ARRAY_TASK_ID}]}
    echo "Input video: $SAMPLE"
    echo "--------"

    # Reencode video if required
    # following SLEAP's recommendations
    # https://sleap.ai/help.html#does-my-data-need-to-be-in-a-particular-format
    if [ "$flag_reencode_input_videos"=true ] ; then 
        echo "Rencoding ...."

        # path to reencoded video
        filename_no_ext="$(basename "$SAMPLE" | sed 's/\(.*\)\..*/\1/')" # filename without extension
        REENCODED_VIDEO_PATH="$REENCODED_VIDEOS_SUBDIR/$filename_no_ext"_RE.$reencoded_extension

        ffmpeg -version  # print version to logs
        ffmpeg -y -i "$SAMPLE" \
        -c:v libx264 \
        -pix_fmt yuv420p \
        -preset superfast \
        -crf 15 \
        $REENCODED_VIDEO_PATH


        echo "Reencoded video: $REENCODED_VIDEO_PATH"
        echo "--------"
        FRAME_EXTRACTION_INPUT_VIDEO=$REENCODED_VIDEO_PATH
    else
        echo "Skipping video reencoding..."
        FRAME_EXTRACTION_INPUT_VIDEO=$SAMPLE
    fi

    # Run frame extraction algorithm on video
    python $SCRIPT_DIR/extract_frames_to_label_w_sleap.py \
    $FRAME_EXTRACTION_INPUT_VIDEO \
    --output_path $OUTPUT_DIR \
    --output_subdir $OUTPUT_SUBDIR \
    --video_extensions $PARAM_VIDEO_EXT \
    --initial_samples $PARAM_INI_SAMPLES \
    --scale $PARAM_SCALE \
    --n_components $PARAM_N_COMPONENTS \
    --n_clusters $PARAM_N_CLUSTERS \
    --per_cluster $PARAM_PER_CLUSTER \
    --compute_features_per_video

    if [ "$?" -ne 0 ]; then
        echo "Frame extraction failed! Please check .err log"
    else
        echo "Frames extracted from video: $FRAME_EXTRACTION_INPUT_VIDEO"
    fi
    echo "--------"

    # Reencoded videos log
    # copy .err file to go with reencoded video too if required
    # filename: {reencoded video name}.{slurm_array}.{slurm_job_id} 
    # TODO: make a nicer log
    if [ "$flag_reencode_input_videos"=true ] ; then 
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
