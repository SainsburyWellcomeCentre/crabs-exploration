#!/bin/bash

#SBATCH -p cpu # partition 
#SBATCH -N 1   # number of nodes
#SBATCH --ntasks-per-node 2
#SBATCH --mem 32G
#SBATCH -t 0-20:00 # time (D-HH:MM)
#SBATCH -o slurm.%A.%N.out
#SBATCH -e slurm.%A.%N.err
#SBATCH --mail-type=ALL
#SBATCH --mail-user=s.minano@ucl.ac.uk

# TODO: make an array job?

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

OUTPUT_DIR=''

LOGS_DIR=''


# ---------------------------
# uv setup
# ---------------------------
module load uv

# set uv cache dir to /ceph/scratch/sminano
# (should be faster than /nfs/nhome/live/sminano/.cache/uv and
# gets purged regularly)
export UV_CACHE_DIR=/ceph/scratch/sminano/uv-cache
# The uv cache and the env are on different filesystems (ceph vs tmpfs)
# so we set link mode to copy across the necessary files,
# instead of symlinking (which would not work across filesystems)
export UV_LINK_MODE=copy
export UV_HTTP_TIMEOUT=120  # seconds

# --------------------
# Extract frames
# ------------------

# ---- Extract last frame from each mp4 ----
uv run --with sleap-io --with imageio python -c "
import sleap_io as sio
import imageio.v3 as iio
from pathlib import Path

loops_dir = Path('$LOOPS_DIR')
output_dir = Path('$OUTPUT_DIR')

for video_path in sorted(loops_dir.glob('*.mp4')):

    video = sio.load_video(str(video_path))
    last_frame = video[-1]

    out = output_dir / f'{video_path.stem}_last_frame.png'
    iio.imwrite(str(out), last_frame)
    print(f'Saved {out}')
"