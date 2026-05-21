#!/bin/bash

#SBATCH -p cpu # partition (cpu or gpu if needed)
#SBATCH -N 1   # number of nodes
#SBATCH --ntasks-per-node 1
#SBATCH --mem 16G
#SBATCH -t 0-04:00 # time (D-HH:MM)
#SBATCH -o slurm.%j.%N.out
#SBATCH -e slurm.%j.%N.err
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
ZARR_STORE="/ceph/zoo/processed/CrabField/ramalhete_2023/CrabTracks/CrabTracks-slurm2478780-2478861-2489356.zarr"

# Output directories
# Note: The Python scripts will create burrow_prompts/coords_<ts>/ and 
# burrow_prompts/frames_<ts>/ respectively (because they append the 
# timestamp <ts> to the path passed).
OUTPUT_DIR="/ceph/zoo/users/sminano/burrow_prompts_slurm_$SLURM_JOB_ID"
OUTPUT_DIR_COORDS="$OUTPUT_DIR/coords" # will be timestamped
OUTPUT_DIR_FRAMES="$OUTPUT_DIR/frames" # will be timestamped

# Version of the codebase: branch (or tag/commit) to fetch the script from
GIT_REPO=SainsburyWellcomeCentre/crabs-exploration
GIT_BRANCH=smg/segment-burrows

# Data grouping for prompt coordinates
DATA_GROUPING_COORD_PROMPTS="video"  # "video" or "date"

# ------------------
# Get script paths
# ------------------

# log the corresponding git commit
GIT_COMMIT_ID=$(git ls-remote "https://github.com/$GIT_REPO.git" "$GIT_BRANCH" | cut -f1)

# Script URLs on GitHub
SCRIPT_COORD_PROMPTS_URL="https://raw.githubusercontent.com/$GIT_REPO/$GIT_COMMIT_ID/crabs/utils/compute_burrow_prompt_coords.py"
SCRIPT_FRAME_PROMPTS_URL="https://raw.githubusercontent.com/$GIT_REPO/$GIT_COMMIT_ID/crabs/utils/compute_burrow_prompt_frames.py"



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
echo "Git commit ID: $GIT_COMMIT_ID"
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

# Track the resolved (timestamped) output dirs printed by each script
# via lines of the form: "Output written to <path>."
RESOLVED_OUTPUT_DIRS=()

# run command
# - we use --reinstall flag to force uv to rebuild
#  the environment without wiping the cache directory 
#  (this is useful because interrupted jobs may lead to
#  corrupted environments that uv otherwise would use,
#  so we force a fresh environment definition here)
# - prepend /usr/bin/time -v to log maxRSS

# create temporary file to capture timestamped output directory
COORDS_LOG=$(mktemp)

echo "Computing prompt coordinates..."
uv run --reinstall "$SCRIPT_COORD_PROMPTS_URL" \
    "$ZARR_STORE" \
    "$OUTPUT_DIR_COORDS" \
    $DATA_GROUPING_FLAG \
    --save-html-figure 2>&1 | tee "$COORDS_LOG"

RESOLVED_OUTPUT_DIRS+=("$(grep -oP '(?<=Output written to )[^.]+' "$COORDS_LOG")")
echo "Prompt coordinates saved at ${RESOLVED_OUTPUT_DIRS[-1]}"

# delete temporary file
rm -f "$COORDS_LOG"

# -----------------------------------------
# Run script to compute prompt frames
# -----------------------------------------

# create temporary file to capture timestamped output directory
FRAMES_LOG=$(mktemp)

# run command
echo "Computing prompt frames..."
uv run --reinstall "$SCRIPT_FRAME_PROMPTS_URL" \
    "$ZARR_STORE" \
    "$OUTPUT_DIR_FRAMES" \
    --save-html-figure 2>&1 | tee "$FRAMES_LOG"

# extract timestamped output dir
RESOLVED_OUTPUT_DIRS+=("$(grep -oP '(?<=Output written to )[^.]+' "$FRAMES_LOG")")
echo "Prompt frames saved at ${RESOLVED_OUTPUT_DIRS[-1]}"

# delete temporary file
rm -f "$FRAMES_LOG"

# --------------------------------------
# Save a copy of the logs under parent output dir
# -------------------------------------

LOG_DIR="$OUTPUT_DIR/logs"
mkdir -p "$LOG_DIR"
cp slurm.$SLURM_JOB_ID.$SLURMD_NODENAME.{err,out} "$LOG_DIR"
chmod 444 "$LOG_DIR"/slurm.$SLURM_JOB_ID.$SLURMD_NODENAME.{err,out}

rm slurm.$SLURM_JOB_ID.$SLURMD_NODENAME.{err,out}
