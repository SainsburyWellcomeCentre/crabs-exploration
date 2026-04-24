#!/bin/bash

#SBATCH -p cpu # partition 
#SBATCH -N 1   # number of nodes
#SBATCH --ntasks-per-node 1
#SBATCH --mem 32G
#SBATCH -t 0-20:00 # time (D-HH:MM)
#SBATCH -o slurm_array.%A-%a.%N.out
#SBATCH -e slurm_array.%A-%a.%N.err
#SBATCH --mail-type=ALL
#SBATCH --mail-user=s.minano@ucl.ac.uk
#SBATCH --array=0-233%25
#SBATCH --constraint=AVX2

set -e
set -u
set -o pipefail



# A script that goes through the clips in "Loops" and extracts
# the last frame from each clip. The last frame per clip should 
# correspond to the 'end escape' frame.
#
# The script assumes the virtual environment has been setup already.
# It is best to built the venv on a node with the same CPU architecture
# as the one where the job runs.
# To do this, run on an interactive compute node session:
#   srun -p cpu --pty bash   # get an interactive compute node session
#   module load uv
#   export UV_CACHE_DIR=/ceph/scratch/sminano/uv-cache # optional 
#   export UV_LINK_MODE=copy # optional 
#   uv venv --no-project /ceph/zoo/users/sminano/envs/sleap-io-env
#   uv pip install --python /ceph/zoo/users/sminano/envs/sleap-io-env/bin/python sleap-io
#   exit        # exit the interactive session


# ---------------------
# Define variables
# ----------------------
LOOPS_DIR="/ceph/zoo/processed/CrabField/ramalhete_2023/Loops"

OUTPUT_DIR="/ceph/zoo/users/sminano/crab_loops_end_frames_slurm$SLURM_ARRAY_JOB_ID"

LOG_DIR=$OUTPUT_DIR/logs
mkdir -p "$LOG_DIR"  # create full path if it doesnt exist

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


# --------------------
# Extract frames
# ------------------

# Pass this task's video path to Python call
VIDEO_PATH="${LIST_VIDEOS[$SLURM_ARRAY_TASK_ID]}"

# Activate the predefined environment
source /ceph/zoo/users/sminano/envs/sleap-io-env/bin/activate

python -c "
import av
import imageio.v3 as iio
from pathlib import Path

video_path = Path('$VIDEO_PATH')
output_path = Path('$OUTPUT_DIR') / f'{video_path.stem}_first_frame.png'

with av.open(video_path) as container:
    frame = next(container.decode(video=0))
    img = frame.to_ndarray(format="rgb24")

iio.imwrite(output_path, img)
print(f'Saved {out}')
"

# ------------------
# Copy logs to LOG_DIR
# -------------------
mv "slurm_array.$SLURM_ARRAY_JOB_ID-$SLURM_ARRAY_TASK_ID.$SLURMD_NODENAME."{err,out} "$LOG_DIR"

# make logs read only
chmod 444 "$LOG_DIR/slurm_array.$SLURM_ARRAY_JOB_ID-$SLURM_ARRAY_TASK_ID.$SLURMD_NODENAME."{err,out}
