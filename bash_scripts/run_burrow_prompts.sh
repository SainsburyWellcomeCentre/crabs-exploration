#!/bin/bash

#SBATCH -p gpu # partition (or gpu if needed)
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

# ---------------------
# Define variables
# ----------------------
ZARR_STORE="/ceph/zoo/users/sminano/CrabTracks-slurmXXXXXXX.zarr"
OUTPUT_DIR="/ceph/zoo/users/sminano/burrow_prompts"

# Version of the codebase: branch (or tag/commit) to fetch the script from
GIT_BRANCH=main

# Script URL on GitHub
SCRIPT_URL="https://raw.githubusercontent.com/SainsburyWellcomeCentre/crabs-exploration/$GIT_BRANCH/crabs/utils/create_burrow_prompts.py"

# location of SLURM logs (resolved at script-resolution time, after the
# create_burrow_prompts.py run timestamps and creates ${OUTPUT_DIR}_<ts>)
LOG_PARENT_DIR=$(dirname "$OUTPUT_DIR")


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
echo "Script URL: $SCRIPT_URL"
echo "zarr_store: $ZARR_STORE"
echo "output_dir: $OUTPUT_DIR"
echo "-----"


# -------------------------
# Run script
# -------------------------
# uv run resolves the PEP 723 inline dependencies declared in the script
# and runs it in an ephemeral environment.
uv run "$SCRIPT_URL" \
    "$ZARR_STORE" \
    "$OUTPUT_DIR" \
    --save-html-figure


# ------------------
# Move logs into the latest output dir
# ------------------
# create_burrow_prompts.py creates "${OUTPUT_DIR}_<timestamp>"; pick the
# most recent one matching that prefix to move logs into.
LATEST_OUTPUT_DIR=$(ls -td "${OUTPUT_DIR}"_* 2>/dev/null | head -n 1)
if [[ -n "$LATEST_OUTPUT_DIR" ]]; then
    LOG_DIR="$LATEST_OUTPUT_DIR/logs"
    mkdir -p "$LOG_DIR"
    mv slurm.$SLURMD_NODENAME.$SLURM_JOB_ID.{err,out} "$LOG_DIR"
    chmod 444 "$LOG_DIR"/slurm.$SLURMD_NODENAME.$SLURM_JOB_ID.{err,out}
fi
