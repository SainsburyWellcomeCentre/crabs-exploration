#!/bin/bash

#SBATCH -p cpu # partition 
#SBATCH -N 1   # number of nodes
#SBATCH --ntasks-per-node 2
#SBATCH --mem 32G
#SBATCH -t 0-20:00 # time (D-HH:MM)
#SBATCH -o slurm_array.%A-%a.%N.out
#SBATCH -e slurm_array.%A-%a.%N.err
#SBATCH --mail-type=ALL
#SBATCH --mail-user=s.minano@ucl.ac.uk
#SBATCH --array=0-25%234 

set -e
set -u
set -o pipefail



# A script that goes through the clips in "Loops" and extracts
# the last frame from each clip. The last frame per clip should 
# correspond to the 'end escape' frame

# ---------------------
# Define variables
# ----------------------
LOOPS_DIR="/ceph/zoo/processed/CrabField/ramalhete_2023/Loops"

OUTPUT_DIR="/ceph/zoo/users/sminano/crab_loops_end_frames_$(date +%Y%m%d_%H%M%S)"

LOGS_DIR=$OUTPUT_DIR/logs
mkdir -p $LOG_DIR  # create if it doesnt exist

# --------------------
# Check inputs
# --------------------
# Check number of video files in LOOPS_DIR matches max SLURM_ARRAY_TASK_COUNT
# if not, exit

# Build ordered list of files
mapfile -t LIST_VIDEOS < <(find "$LOOPS_DIR" -name "*.mp4" | sort)
N_VIDEOS=${#LIST_VIDEOS[@]}

if [[ $SLURM_ARRAY_TASK_COUNT -ne $N_VIDEOS ]]; then
    echo "The number of array tasks does not match the number of videos in the input directory."
    echo "  Array tasks:    $SLURM_ARRAY_TASK_COUNT"
    echo "  Unique videos:  $N_VIDEOS"
    exit 1
fi


# ---------------------------
# uv setup
# ---------------------------
# module load uv

# # set uv cache dir to /ceph/scratch/sminano
# # (should be faster than /nfs/nhome/live/sminano/.cache/uv and
# # gets purged regularly)
# export UV_CACHE_DIR=/ceph/scratch/sminano/uv-cache
# # The uv cache and the env are on different filesystems (ceph vs tmpfs)
# # so we set link mode to copy across the necessary files,
# # instead of symlinking (which would not work across filesystems)
# export UV_LINK_MODE=copy
# export UV_HTTP_TIMEOUT=120  # seconds

# --------------------
# Extract frames
# ------------------

# Pass this task's video path to Python call
VIDEO_PATH="${LIST_VIDEOS[$SLURM_ARRAY_TASK_ID]}"

source /ceph/zoo/users/sminano/envs/sleap-io-env/bin/activate

python -c "
import sys
import sleap_io as sio
import imageio.v3 as iio
from pathlib import Path

video_path = Path('$VIDEO_PATH')
video = sio.load_video(str(video_path))

frame_idx = video.shape[0] - 1
last_frame = video[frame_idx]

out = Path('$OUTPUT_DIR') / f'{video_path.stem}_frame_{frame_idx:0>8d}.png'
iio.imwrite(str(out), last_frame)

print(f'Saved {out}')
"

# ------------------
# Copy logs to LOG_DIR
# -------------------
mv slurm_array.$SLURM_ARRAY_JOB_ID-$SLURM_ARRAY_TASK_ID.$SLURMD_NODENAME.{err,out} $LOG_DIR

# make logs read only
chmod 444 $LOG_DIR/slurm_array.$SLURM_ARRAY_JOB_ID-$SLURM_ARRAY_TASK_ID.$SLURMD_NODENAME.{err,out}
