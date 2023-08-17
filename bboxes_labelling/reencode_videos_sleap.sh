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
#SBATCH --array=0-7%4

# load modules
module load SLEAP


# input/output data dirs
INPUT_DATA_LIST=(
    "/ceph/zoo/raw/CrabField/ramalhete_2023/09.08.2023-Day2/09.08.2023-01-Left.MOV"
    # "/ceph/zoo/raw/CrabField/ramalhete_2023/09.08.2023-Day2/09.08.2023-01-Right.mov" #
    # "/ceph/zoo/raw/CrabField/ramalhete_2023/09.08.2023-Day2/09.08.2023-02-Left.mov" #
    # "/ceph/zoo/raw/CrabField/ramalhete_2023/09.08.2023-Day2/09.08.2023-02-Right.MOV" #
    "/ceph/zoo/raw/CrabField/ramalhete_2023/09.08.2023-Day2/09.08.2023-03-Left.MOV"
    "/ceph/zoo/raw/CrabField/ramalhete_2023/09.08.2023-Day2/09.08.2023-04-Left.MOV"
    "/ceph/zoo/raw/CrabField/ramalhete_2023/09.08.2023-Day2/09.08.2023-04-Right.MOV"
    "/ceph/zoo/raw/CrabField/ramalhete_2023/10.08.2023-Day3/10.08.2023-01-Left.mov"
    "/ceph/zoo/raw/CrabField/ramalhete_2023/10.08.2023-Day3/10.08.2023-01-Right.mov"
    "/ceph/zoo/raw/CrabField/ramalhete_2023/10.08.2023-Day3/10.08.2023-02-Left.MOV"
    "/ceph/zoo/raw/CrabField/ramalhete_2023/10.08.2023-Day3/10.08.2023-02-Right.MOV"

)

OUTPUT_DIR=/ceph/zoo/users/sminano/crabs_reencoding
OUTPUT_SUBDIR="20230816_ramalhete2023_day2"

echo "Input data:"
for i in "${INPUT_DATA_LIST[@]}"
do
    echo "$i"
done
echo "Output dir: $OUTPUT_DIR/$OUTPUT_SUBDIR"
echo "--------"

# Check len(list of input data) matches max SLURM_ARRAY_TASK_COUNT
# if not, exit
if [[ $SLURM_ARRAY_TASK_COUNT -ne ${#INPUT_DATA_LIST[@]} ]]; then
    echo "The number of array tasks does not match the number of inputs"
    exit 1
fi

# reencode input videos following SLEAP's recommendations
# https://sleap.ai/help.html#does-my-data-need-to-be-in-a-particular-format
for i in {1..${SLURM_ARRAY_TASK_COUNT}}
do
    FILEPATH=${INPUT_DATA_LIST[${SLURM_ARRAY_TASK_ID}]}
    filename_no_ext="$(basename "$FILEPATH" | sed 's/\(.*\)\..*/\1/')"
    echo "Input video: $FILEPATH"

    ffmpeg -y -i "$FILEPATH" \
    -c:v libx264 \
    -pix_fmt yuv420p \
    -preset superfast \
    -crf 15 \
    "$OUTPUT_DIR/$OUTPUT_SUBDIR/$filename_no_ext.mp4"

    echo "Reencoded video: $OUTPUT_DIR/$OUTPUT_SUBDIR/$filename_no_ext.mp4"
    echo "---"
done

# apple encoder settings?
# https://ottverse.com/ffmpeg-convert-to-apple-prores-422-4444-hq/#:~:text=Encoding%20Apple%20ProRes%20422%20HQ%20using%20FFmpeg,-The%20commandline%20for&text=If%20you%20see%20the%20commandline,get%20a%20hang%20of%20it.

# ffmpeg -y -i "$FILEPATH" \
# -c:v prores_ks \
# -profile:v 3 \
# -vendor appl \
# -pix_fmt yuv422p10le \
# -preset superfast \
# -crf 15 \
# "$OUTPUT_DIR/$OUTPUT_SUBDIR/$filename_no_ext.MOV"