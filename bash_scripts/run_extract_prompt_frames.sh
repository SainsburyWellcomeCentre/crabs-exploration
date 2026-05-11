#!/bin/bash

#SBATCH -p cpu                    # partition
#SBATCH -N 1                      # number of nodes
#SBATCH --ntasks-per-node 1       # number of tasks per node
#SBATCH --mem 8G                  # memory pool for all cores
#SBATCH -t 0-04:00                # time (D-HH:MM)
#SBATCH -o slurm_array.%N.%A-%a.out
#SBATCH -e slurm_array.%N.%A-%a.err
#SBATCH --mail-type=ALL
#SBATCH --mail-user=s.minano@ucl.ac.uk
# Update N to (number of unique clips in INPUT_FRAMES_CSV) - 1.
# To get number of unique clips:
#   awk -F',' 'NR>2 && $1 !~ /^#/ {print $1}' "$INPUT_FRAMES_CSV" | sort -u | wc -l
#SBATCH --array=0-119%10

set -e
set -u
set -o pipefail

# ---------------------
# Define variables
# ---------------------
INPUT_CLIPS_DIR="/ceph/zoo/users/sminano/loop_clips"
INPUT_FRAMES_CSV="/ceph/zoo/users/sminano/burrow_prompts_XXXXXXXX_XXXXXX/frames_to_extract.csv"
OUTPUT_DIR="/ceph/zoo/users/sminano/burrow_prompt_frames"


# ---------------------
# Derive list of unique clip names from the CSV
# (with nR>2, it skips the leading # comment line and the header row)
# ---------------------
mapfile -t CLIPS_LIST < <(
    awk -F',' 'NR>2 && $1 !~ /^#/ {print $1}' "$FRAMES_CSV" | sort -u
)

# Check len(CLIPS_LIST) matches SLURM_ARRAY_TASK_COUNT
if [[ $SLURM_ARRAY_TASK_COUNT -ne ${#CLIPS_LIST[@]} ]]; then
    echo "SLURM_ARRAY_TASK_COUNT (${SLURM_ARRAY_TASK_COUNT}) does not match" \
         "the number of unique clips in the CSV (${#CLIPS_LIST[@]})"
    exit 1
fi


# -------------------------------
# Create output directory
# All tasks in the array write to the same directory, identified by
# the SLURM array job ID.
# -------------------------------
OUTPUT_DIR_JOB="${OUTPUT_DIR}_${SLURM_ARRAY_JOB_ID}"
mkdir -p "$OUTPUT_DIR_JOB"


# -------------------------
# Log arguments
# -------------------------
echo "Input clips directory:  $INPUT_CLIPS_DIR"
echo "Path to csv with frames to extract: $INPUT_FRAMES_CSV"
echo "Output directory: $OUTPUT_DIR_JOB"
echo "-----"

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

# ----------------
# warmup uv cache
# ----------------
# A lock is a coordination primitive: only one process at a time can "hold" it.
# Other processes that try to acquire it block (wait) until the holder releases it.
# It's how you serialize access to a shared resource across independent processes
# that otherwise can't talk to each other.
#
# flock (file-lock) uses a file as the rendezvous point.
#
# A file descriptor (FD) is a small integer the kernel hands you when you open a file
# — it's your handle to that open file.
#
# What happens below?
# All 10 concurrent array tasks hit this block.
# One wins the lock and does the slow cold-cache download.
# The other 9 wait. When the winner finishes, the next acquires the lock,
# finds a warm cache, returns immediately, releases — and so on.
# No two tasks ever download in parallel, no cache corruption, no wasted bandwidth.
#
# The lock file itself (.warmup.lock) just exists as an anchor — nothing is ever written to it.
# You can leave it on disk between runs.

# create cache dir if it does not exist
mkdir -p "$UV_CACHE_DIR"

# put a lock file (.warmup.lock) inside cache dir.
# once the lock is held, run the uv cache warmup
# First task to arrive populates the uv cache;
# later tasks block on flock, then run the same command when released and
# find a warm cache (instant).
flock -x "$UV_CACHE_DIR/.warmup.lock" \
    uv run --python 3.11 --with av -- python -c "import av" >/dev/null


# -------------------------------------
# Get clip for this job in the array
# -------------------------------------
CLIP_NAME="${CLIPS_LIST[$SLURM_ARRAY_TASK_ID]}"
CLIP_NAME_NO_EXT="${CLIP_NAME%.mp4}"
CLIP_PATH="$INPUT_CLIPS_DIR/$CLIP_NAME"

if [[ ! -f "$CLIP_PATH" ]]; then
    echo "WARNING: file not found at $CLIP_PATH — skipping"
    exit 0
fi

echo "Extracting frames for clip $CLIP_NAME ..."

# ----------------------------------
# Get frame indices for this clip
# ----------------------------------
# Get list of frame indices to extract for this clip AND
# sort in ascending order (!! not by count)
mapfile -t FRAME_IDCS < <(
    awk -F',' -v clip="$CLIP_NAME" \
        'NR>2 && $1 !~ /^#/ && $1==clip {print $2+0}' \
        "$INPUT_FRAMES_CSV" | sort -n
)

echo "Frames to extract (n=${#FRAME_IDCS[@]}): $(IFS=,; echo "${FRAME_IDCS[*]}")"

# -----------------------------------
# Extract all frames for this clip
# -----------------------------------

# Run ffmpeg command to extract frames in decode order
# (we assume input video is reencoded such that frames are reliably
# seekable)
# -------------------------------------
# run Python on the script written between <<'PYEOF' and PYEOF
uv run --python 3.11 --with av - "$CLIP_PATH" "$OUTPUT_DIR_JOB" "$CLIP_NAME_NO_EXT" "${FRAME_IDCS[@]}" <<'PYEOF'
import sys
from fractions import Fraction
from pathlib import Path

import av

clip_path = sys.argv[1]
output_dir = Path(sys.argv[2])
clip_name_no_ext = sys.argv[3]
frame_idcs = [int(x) for x in sys.argv[4:]]

with av.open(clip_path) as container:
    stream = container.streams.video[0]
    fps = stream.codec_context.framerate  # Fraction

    # Loop thru frame idcs to extract
    for target_idx in frame_idcs:

        # Go to nearest keyframe **before** target (time in microseconds)
        container.seek(int(target_idx / float(fps) * 1e6))

        # Decode from keyframe until we get a frame with PTS >= target
        target_pts = None # initialise
        for frame in container.decode(stream):
            # Compute target_pts from data in first frame
            # (we need to get frame.time_base from first frame)
            if target_pts is None:
                target_pts = int(Fraction(target_idx) / fps / frame.time_base)

            # Throw an error if PTS for this frame not defined
            # (possible in some containers)
            if frame.pts is None:
                raise ValueError(f"Frame at index {target_idx} has no PTS")

            # Compare current and target PTS values (both are integers)
            if frame.pts >= target_pts:
                out_path = (
                    output_dir
                    / f"{clip_name_no_ext}_frame_{target_idx:08d}.png"
                )
                # frame is an av.VideoFrame; .to_image returns PIL.Image.Image
                frame.to_image().save(out_path)
                break
PYEOF
# -------------------------------------

echo "Extracted ${#FRAME_IDCS[@]} frames from $CLIP_NAME to $OUTPUT_DIR_JOB"


# ----------------------------------------
# Move SLURM logs into the output directory
# ----------------------------------------
# Create slurm directory
LOG_DIR="$OUTPUT_DIR_JOB/logs"
mkdir -p "$LOG_DIR"

mv "slurm_array.$SLURMD_NODENAME.$SLURM_ARRAY_JOB_ID-$SLURM_ARRAY_TASK_ID.out" \
   "$LOG_DIR"
mv "slurm_array.$SLURMD_NODENAME.$SLURM_ARRAY_JOB_ID-$SLURM_ARRAY_TASK_ID.err" \
   "$LOG_DIR"

# make read-only
chmod 444 "$LOG_DIR"/slurm_array.*
