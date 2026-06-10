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

# Version of the codebase: branch (or tag/commit) to install the package from
GIT_REPO=SainsburyWellcomeCentre/crabs-exploration
GIT_BRANCH=smg/segment-burrows

# Data grouping for prompt coordinates
DATA_GROUPING_COORD_PROMPTS="video"  # "video" or "date"

# ------------------
# Resolve git commit
# ------------------
# Resolve (and log) the commit the branch points to, so we can pin the
# install below to it and the logged commit is exactly what runs.
GIT_COMMIT_ID=$(git ls-remote "https://github.com/$GIT_REPO.git" "$GIT_BRANCH" | cut -f1)



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


# ---------------------------------------------
# Create environment and install crabs package
# ---------------------------------------------
# compute-burrow-prompt-coords and compute-burrow-prompt-frames are entry
# points of the crabs package, so we install crabs into a per-job virtual
# environment and call the commands from it. We include the heavy, opt-in
# 'burrows' extra (plotly, datashader, scikit-image, pillow), which these
# commands need for the histogram/peak computation and the
# --save-html-figure output. The install is pinned to the resolved commit.
ENV_NAME=crabs-burrow-prompts-$SLURM_JOB_ID
ENV_PREFIX=$TMPDIR/$ENV_NAME

uv venv "$ENV_PREFIX" --python 3.12
source "$ENV_PREFIX/bin/activate"

uv pip install "crabs[burrows] @ git+https://github.com/$GIT_REPO.git@$GIT_COMMIT_ID"

# log python location and installed crabs version
which python
uv pip show crabs


# -------------------------
# Log arguments
# -------------------------
echo "Git branch: $GIT_BRANCH"
echo "Git commit ID: $GIT_COMMIT_ID"
echo "zarr_store: $ZARR_STORE"
echo "output_dir: $OUTPUT_DIR"
echo "output_dir coordinates: $OUTPUT_DIR_COORDS"
echo "output_dir frames: $OUTPUT_DIR_FRAMES"
echo "data grouping strategy for prompt coordinates: by $DATA_GROUPING_COORD_PROMPTS"
echo "-----"


# -----------------------------------------
# Run script to compute prompt coordinates
# -----------------------------------------
# compute-burrow-prompt-coords is a console entry point provided by the
# crabs package installed in the active venv above.

# Determine data grouping for computing burrow hotspots
# if grouping by date, add the "--group-by-pattern" flag
DATA_GROUPING_FLAG=""
if [[ "$DATA_GROUPING_COORD_PROMPTS" == "date" ]]; then
    DATA_GROUPING_FLAG="--group-by-pattern"
fi

# Track the resolved (timestamped) output dirs printed by each script
# via lines of the form: "Output written to <path>."
RESOLVED_OUTPUT_DIRS=()

# run command (the per-job venv with crabs[burrows] is already active)
# - prepend /usr/bin/time -v to log maxRSS if needed

# create temporary file to capture timestamped output directory
COORDS_LOG=$(mktemp)

echo "Computing prompt coordinates..."
compute-burrow-prompt-coords \
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
compute-burrow-prompt-frames \
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


# -----------------------------
# Cleanup
# -----------------------------
deactivate
rm -rf "$ENV_PREFIX"
