#!/bin/bash

#SBATCH -p cpu # partition (cpu or gpu if needed)
#SBATCH -N 1   # number of nodes
#SBATCH --ntasks-per-node 2
#SBATCH --mem 35G
#SBATCH -t 0-04:00 # time (D-HH:MM)
#SBATCH -o slurm.%N.%j.out
#SBATCH -e slurm.%N.%j.err
#SBATCH --mail-type=ALL
#SBATCH --mail-user=s.minano@ucl.ac.uk

set -e
set -u
set -o pipefail

# ---------------
# Description
# --------------
# This script computes the data that is necessary to prompt SAM3 to segment burrows.
# It runs:
# - a script to compute the x,y coordinates for the burrow prompts,
#   and saves the outputs to OUTPUT_DIR_COORDS
# - a script to compute the frames to extract for the burrow prompts 
#   and saves the outputs to OUTPUT_DIR_FRAMES

# ---------------------
# Define variables
# ----------------------
ZARR_STORE="/ceph/zoo/processed/CrabField/ramalhete_2023/CrabTracks/CrabTracks-slurm2492830-slurm2492948.zarr"

# Output directories
# Note: The Python scripts will create burrow_prompts/coords_<ts>/ and 
# burrow_prompts/frames_<ts>/ respectively (because they append the 
# timestamp <ts> to the path passed).
OUTPUT_DIR="/ceph/zoo/users/sminano/burrow_prompts"
OUTPUT_DIR_COORDS="/ceph/zoo/users/sminano/burrow_prompts/coords"
OUTPUT_DIR_FRAMES="/ceph/zoo/users/sminano/burrow_prompts/frames"

# Version of the codebase: branch (or tag/commit) to fetch the script from
GIT_BRANCH=smg/segment-burrows

# Script URLs on GitHub
SCRIPT_COORD_PROMPTS_URL="https://raw.githubusercontent.com/SainsburyWellcomeCentre/crabs-exploration/$GIT_BRANCH/crabs/utils/compute_burrow_prompt_coords.py"
SCRIPT_FRAME_PROMPTS_URL="https://raw.githubusercontent.com/SainsburyWellcomeCentre/crabs-exploration/$GIT_BRANCH/crabs/utils/compute_burrow_prompt_frames.py"

# Data grouping for prompt coordinates
DATA_GROUPING_COORD_PROMPTS="video"  # "video" or "date"

# ---------------------------
# uv configuration
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


# -------------------------
# Log arguments
# -------------------------
echo "Git branch: $GIT_BRANCH"
echo "Script to compute prompt coordinates URL: $SCRIPT_COORD_PROMPTS_URL"
echo "Script to compute prompt frames URL: $SCRIPT_FRAME_PROMPTS_URL"
echo "zarr_store: $ZARR_STORE"
echo "output_dir: $OUTPUT_DIR"
echo "output_dir coordinates: $OUTPUT_DIR_COORDS"
echo "output_dir frames: $OUTPUT_DIR_FRAMES"
echo "data grouping strategy for prompt coordinates: by $DATA_GROUPING_COORD_PROMPTS"
echo "-----"


# -----------------------------------------
# Run script to compute prompt coordinates
# -----------------------------------------
# uv run resolves the PEP 723 inline dependencies declared in the script
# and runs it in an ephemeral environment.

# Determine data grouping for computing burrow hotspots
# if grouping by date, add the "--group-by-pattern" flag
DATA_GROUPING_FLAG=""
if [[ "$DATA_GROUPING_COORD_PROMPTS" == "date" ]]; then
    DATA_GROUPING_FLAG="--group-by-pattern"
fi

# we use --reinstall flag to force uv to rebuild
# the environment without wiping the cache directory 
# (this is useful because interrupted jobs may lead to
# corrupted environments that uv otherwise would use,
# so we force a fresh environment definition here)


# Track the resolved (timestamped) output dirs printed by each script
# via lines of the form: "Output written to <path>."
RESOLVED_OUTPUT_DIRS=()

# create temporary file to capture timestamped output directory
COORDS_LOG=$(mktemp)

# run command
# --save-html-figure \
echo "Computing prompt coordinates..."
/usr/bin/time -v uv run --reinstall "$SCRIPT_COORD_PROMPTS_URL" \
    "$ZARR_STORE" \
    "$OUTPUT_DIR_COORDS" \
    $DATA_GROUPING_FLAG 2>&1 | tee "$COORDS_LOG"

RESOLVED_OUTPUT_DIRS+=("$(grep -oP '(?<=Output written to )[^.]+' "$COORDS_LOG")")
rm -f "$COORDS_LOG"
echo "Prompt coordinates saved at ${RESOLVED_OUTPUT_DIRS[-1]}"

# -----------------------------------------
# Run script to compute prompt frames
# -----------------------------------------

# # create temporary file to capture timestamped output directory
# FRAMES_LOG=$(mktemp)

# # run command
# echo "Computing prompt frames..."
# uv run --reinstall "$SCRIPT_FRAME_PROMPTS_URL" \
#     "$ZARR_STORE" \
#     "$OUTPUT_DIR_FRAMES" \
#     --save-html-figure 2>&1 | tee "$FRAMES_LOG"

# # extract timestamped output dir
# RESOLVED_OUTPUT_DIRS+=("$(grep -oP '(?<=Output written to )[^.]+' "$FRAMES_LOG")")
# rm -f "$FRAMES_LOG"
# echo "Prompt frames saved at ${RESOLVED_OUTPUT_DIRS[-1]}"

# --------------------------------------
# Save a copy of the logs under each resolved output dir
# -------------------------------------
for DIR in "${RESOLVED_OUTPUT_DIRS[@]}"; do
    LOG_DIR="$DIR/logs"
    mkdir -p "$LOG_DIR"
    cp slurm.$SLURMD_NODENAME.$SLURM_JOB_ID.{err,out} "$LOG_DIR"
    chmod 444 "$LOG_DIR"/slurm.$SLURMD_NODENAME.$SLURM_JOB_ID.{err,out}
done
rm slurm.$SLURMD_NODENAME.$SLURM_JOB_ID.{err,out}
