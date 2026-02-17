#!/bin/bash

#SBATCH -p gpu # partition (or gpu if needed)
#SBATCH -N 1   # number of nodes
#SBATCH --ntasks-per-node 2
#SBATCH --mem 8G
#SBATCH -t 0-20:00 # time (D-HH:MM)
#SBATCH -o slurm_array.%A-%a.%N.out
#SBATCH -e slurm_array.%A-%a.%N.err
#SBATCH --mail-type=ALL
#SBATCH --mail-user=s.minano@ucl.ac.uk

# NOTE: we assume this script is run from the same directory the
# failed log files for store_1 are at

set -e
set -u
set -o pipefail

# ---------------------
# Define variables
# ----------------------
# Set paths to the zarr stores to merge
# We will integrate the results of store_2 into store_1
STORE_1="/path/to/store_1/CrabTracks-slurm1234.zarr"  # final store
STORE_2="/path/to/store_2/CrabTracks-slurm5678.zarr"  # store to merge into final one


# ---------------------
# Extract SLURM job IDs
# ----------------------
SLURM_ARRAY_JOB_ID_1=$(basename "$STORE_1" | sed -n 's/.*slurm\([0-9]\+\).zarr/\1/p')
SLURM_ARRAY_JOB_ID_2=$(basename "$STORE_2" | sed -n 's/.*slurm\([0-9]\+\).zarr/\1/p')


# -----------------------------------------
# Move failed logs from first run (store_1)
# -----------------------------------------
# Failed logs may be at the directory from which this script is ran.
# First check if they exist, and if so move them to a filed logs directory
# under store_1

if ls slurm_array."${SLURM_ARRAY_JOB_ID_1}"-*.{err,out} &> /dev/null; then
    # Create failed logs dir
    FAILED_LOGS_DIR="$STORE_1/logs_failed"
    mkdir -p "$FAILED_LOGS_DIR"

    # Move logs across
    for f in slurm_array."${SLURM_ARRAY_JOB_ID_1}"-*.{err,out}; do
        [ -f "$f" ] && mv "$f" "$FAILED_LOGS_DIR/"
    done
fi

# ------------------------------
# Rename the merged zarr store
# ------------------------------
# to include the SLURM job ID from the second run
MERGED_ZARR_DIR="CrabTracks-slurm${SLURM_ARRAY_JOB_ID_1}-slurm${SLURM_ARRAY_JOB_ID_2}.zarr"
MERGED_ZARR_PATH="$(dirname "$STORE_1")/$MERGED_ZARR_DIR"
mv "$STORE_1" "$MERGED_ZARR_PATH"

# ------------------------------
# Move video directories across
# -------------------------------
# Move video directories (groups) from store_2 to MERGED_store
# NOTE: the glob pattern "$STORE_2"/*/ in the for loop only matches directories
for dir in "$STORE_2"/*/; do
    # Skip if directory is logs or root zarr.json
    if [[ "$(basename "$dir")" == "logs" ]]; then
        continue
    fi
    # Only move directories (not files like zarr.json)
    if [ -d "$dir" ]; then
        mv "$dir" "$MERGED_ZARR_PATH/"
    fi
done

# ------------------------------------------
# Move logs from store_2 across
# ------------------------------------------
# Move logs from store_2/logs to MERGED_store/logs
# NOTE: If the logs directory exists but is empty,
# the * won't expand and mv will fail.
if [ -d "$STORE_2/logs" ]; then
    mv "$STORE_2/logs/"* "$MERGED_ZARR_PATH/logs/"
fi

# CAUTION:
# Do NOT move the root zarr.json from store_2 to MERGED_store,
# since that would overwrite it completely. We need to
# update the zarr.json file with the new videos, we do that
# via Python

# ---------------------------------------------------
# Update zarr.json in MERGED_store to include all videos
# ---------------------------------------------------
# Update metadata JSON file in MERGED_store
python3 -c "
import sys, zarr
zarr.consolidate_metadata(sys.argv[1])
" "$MERGED_ZARR_PATH"


# ------------------------------
# Delete store_2
# ------------------------------
rm -r "${STORE_2:?STORE_2 is not set}"

echo "Merge complete. Final zarr store: $MERGED_ZARR_PATH"
