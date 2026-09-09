#!/bin/bash

#SBATCH -p cpu                    # partition
#SBATCH -N 1                      # number of nodes
#SBATCH --ntasks-per-node 1       # number of tasks per node
#SBATCH --mem 8G                  # memory pool for all cores
#SBATCH -t 0-08:00                # time (D-HH:MM)
#SBATCH -o slurm_array.%A-%a.%N.out
#SBATCH -e slurm_array.%A-%a.%N.err
#SBATCH --mail-type=ALL
#SBATCH --mail-user=s.minano@ucl.ac.uk
# Update N to (number of unique videos in INPUT_FRAMES_CSV) - 1.
# To get number of unique videos:
#   awk -F',' 'NR>2 && $1 !~ /^#/ {print $3}' "$INPUT_FRAMES_CSV" | sort -u | wc -l
#SBATCH --array=0-26%10

set -e
set -u
set -o pipefail

# ---------------------
# Define variables
# ---------------------
INPUT_CLIPS_DIR="/ceph/zoo/processed/CrabField/ramalhete_2023/Loops"
INPUT_FRAMES_CSV="/ceph/zoo/users/sminano/burrow_prompts_slurm_3012602/frames_20260519_121017/frames_to_extract.csv"

# Pixelwise reduction to apply across the selected frames of each video.
# One of: min | max | mean
# (all stream frame-by-frame, so memory stays at one frame per clip)
REDUCTION="min"


OUTPUT_DIR="/ceph/zoo/users/sminano/burrow_${REDUCTION}_image_slurm_${SLURM_ARRAY_JOB_ID}"

# -----------------------------------------------------------------
# Derive list of unique source-video names from the CSV (column 3)
# (with NR>2, it skips the leading # comment line and the header row)
# -----------------------------------------------------------------
mapfile -t VIDEOS_LIST < <(
    awk -F',' 'NR>2 && $1 !~ /^#/ {print $3}' "$INPUT_FRAMES_CSV" | sort -u
)


# --------------
# Checks
# --------------
# Sanity check: column 3 (video_name) must stay consistent with the clip
# filename, i.e. the clip name minus its -LoopNN.<ext> suffix must equal the
# video name minus its extension. Fail loudly if the CSV ever stops matching.
# ([0-9][0-9]* instead of [0-9]+ keeps the awk regex POSIX-portable.)
awk -F',' 'NR>2 && $1 !~ /^#/ {
    c=$1; sub(/-Loop[0-9][0-9]*\.[^.]+$/,"",c)
    v=$3; sub(/\.[^.]+$/,"",v)
    if (c!=v) { print "MISMATCH row "NR": "$1" vs "$3 > "/dev/stderr"; bad=1 }
} END { exit bad?1:0 }' "$INPUT_FRAMES_CSV" \
    || { echo "ERROR: column 3 video_name does not match clip filename"; \
         exit 1; }

# Check len(VIDEOS_LIST) matches SLURM_ARRAY_TASK_COUNT
if [[ $SLURM_ARRAY_TASK_COUNT -ne ${#VIDEOS_LIST[@]} ]]; then
    echo "SLURM_ARRAY_TASK_COUNT (${SLURM_ARRAY_TASK_COUNT}) does not match" \
         "the number of unique videos in the CSV (${#VIDEOS_LIST[@]})"
    exit 1
fi

# Check reduction is valid
case "$REDUCTION" in
    min|max|mean) ;;
    *)
        echo "ERROR: REDUCTION must be one of min|max|mean" \
             "(got '$REDUCTION')"
        exit 1
        ;;
esac

# -------------------------------
# Create output directory
# All tasks in the array write to the same directory, identified by
# the SLURM array job ID.
# -------------------------------
mkdir -p "$OUTPUT_DIR"


# -------------------------
# Log arguments
# -------------------------
echo "Input clips directory:  $INPUT_CLIPS_DIR"
echo "Path to csv with selected frames: $INPUT_FRAMES_CSV"
echo "Output directory: $OUTPUT_DIR"
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
#
# Re dependencies: Pillow is not a hard dependency of av,
# PyAV declares it optional and doesn't pull it in automatically
flock -x "$UV_CACHE_DIR/.warmup.lock" \
    uv run --python 3.11 --with av,pillow,numpy -- python -c "import av, PIL, numpy" >/dev/null


# -------------------------------------
# Get source video for this job in the array
# -------------------------------------
VIDEO_NAME="${VIDEOS_LIST[$SLURM_ARRAY_TASK_ID]}"   # raw column-3 value
VIDEO_NAME_NO_EXT="${VIDEO_NAME%.*}"                # used for log + output

echo "Computing pixelwise $REDUCTION image for video $VIDEO_NAME_NO_EXT ..."

# ----------------------------------
# Get the clips that belong to this video
# (exact column-3 match; the regex consistency check ran once up front)
# ----------------------------------
mapfile -t CLIPS_FOR_VIDEO < <(
    awk -F',' -v v="$VIDEO_NAME" \
        'NR>2 && $1 !~ /^#/ && $3==v {print $1}' \
        "$INPUT_FRAMES_CSV" | sort -u
)

echo "Clips for this video (n=${#CLIPS_FOR_VIDEO[@]}): ${CLIPS_FOR_VIDEO[*]}"

# ----------------------------------
# Build the structured argument list for the Python reducer:
#   <clip_path> <n_frames> <idx> <idx> ...   (one such block per clip)
#
# Frame indices come from column 2 (frame_0idx_in_clip) because we decode
# the clip .mp4 files that exist on disk -- NOT column 4, which indexes the
# source videos. Indices are sorted ascending for seek efficiency.
# A clip whose .mp4 is missing is warned about and skipped, so one missing
# clip does not drop the whole video.
# ----------------------------------
PY_CLIP_ARGS=()
N_FRAMES_TOTAL=0
for CLIP_NAME in "${CLIPS_FOR_VIDEO[@]}"; do
    CLIP_PATH="$INPUT_CLIPS_DIR/$CLIP_NAME"

    if [[ ! -f "$CLIP_PATH" ]]; then
        echo "WARNING: file not found at $CLIP_PATH — skipping this clip"
        continue
    fi

    # Frame indices for this clip, sorted ascending (!! not by count)
    mapfile -t FRAME_IDCS < <(
        awk -F',' -v clip="$CLIP_NAME" \
            'NR>2 && $1 !~ /^#/ && $1==clip {print $2+0}' \
            "$INPUT_FRAMES_CSV" | sort -n
    )

    echo "  $CLIP_NAME: ${#FRAME_IDCS[@]} frames"

    PY_CLIP_ARGS+=("$CLIP_PATH" "${#FRAME_IDCS[@]}" "${FRAME_IDCS[@]}")
    N_FRAMES_TOTAL=$((N_FRAMES_TOTAL + ${#FRAME_IDCS[@]}))
done

if [[ ${#PY_CLIP_ARGS[@]} -eq 0 ]]; then
    echo "WARNING: no existing clips for video $VIDEO_NAME_NO_EXT — skipping"
    exit 0
fi

echo "Total frames to include for $VIDEO_NAME_NO_EXT: $N_FRAMES_TOTAL"

# -----------------------------------------------------
# Compute pixelwise reduction image across all frames
# -----------------------------------------------------

# Run pyav command to decode the requested frames from every clip of this
# video and reduce them pixelwise (min/max/mean) into a single image per
# video (we assume input clips are reencoded such that frames are reliably
# seekable)
# -------------------------------------
# run Python on the script written between <<'PYEOF' and PYEOF
uv run --python 3.11 --with av,pillow,numpy - "$OUTPUT_DIR" "$REDUCTION" "$VIDEO_NAME_NO_EXT" "${PY_CLIP_ARGS[@]}" <<'PYEOF'
import sys
from fractions import Fraction
from pathlib import Path

import av
import numpy as np
from PIL import Image

output_dir = Path(sys.argv[1])
reduction = sys.argv[2]
video_name_no_ext = sys.argv[3]

if reduction not in {"min", "max", "mean"}:
    raise ValueError(f"Unknown reduction '{reduction}'")

# Parse the structured per-clip args passed from bash:
#   <clip_path> <n_frames> <idx> ...   (one such block per clip)
rest = sys.argv[4:]
clips = []  # list of (clip_path, [frame_idcs])
i = 0
while i < len(rest):
    clip_path = rest[i]
    count = int(rest[i + 1])
    clips.append((clip_path, [int(x) for x in rest[i + 2 : i + 2 + count]]))
    i += 2 + count

# Accumulator shared across ALL clips of this video:
# a uint8 running min/max, or a float64 running sum for mean.
# Lazily initialised on the first decoded frame.
acc = None
ref_shape = None
n_used = 0

# Loop thru every clip of this video, folding all its frames into the
# same accumulator
for clip_path, frame_idcs in clips:
    with av.open(clip_path) as container:
        stream = container.streams.video[0]
        fps = stream.codec_context.framerate  # Fraction

        # Loop thru frame idcs to include in the reduction
        for target_idx in frame_idcs:

            # Go to nearest keyframe **before** target (time in microseconds)
            # (assumes constant fps)
            container.seek(int(target_idx / float(fps) * 1e6))

            # Decode from keyframe until we get a frame with PTS >= target
            target_pts = None  # initialise
            for frame in container.decode(stream):
                # Compute target_pts from data in first frame
                # (we need to get frame.time_base from first frame)
                if target_pts is None:
                    target_pts = int(
                        Fraction(target_idx) / fps / frame.time_base
                    )

                # Throw an error if PTS for this frame not defined
                # (possible in some containers)
                if frame.pts is None:
                    raise ValueError(
                        f"Frame {target_idx} of {clip_path} has no PTS"
                    )

                # Compare current and target PTS values (both are integers)
                if frame.pts >= target_pts:
                    # frame is an av.VideoFrame; decode as RGB uint8 array
                    arr = frame.to_ndarray(format="rgb24")

                    # Guard against frames of differing resolution (also
                    # catches clips of the same video differing in size)
                    if ref_shape is None:
                        ref_shape = arr.shape
                    elif arr.shape != ref_shape:
                        raise ValueError(
                            f"Frame {target_idx} of {clip_path} has shape "
                            f"{arr.shape}, expected {ref_shape}"
                        )

                    # Fold this frame into the accumulator in place, so
                    # memory stays at one frame regardless of frame count.
                    if reduction == "min":
                        acc = arr.copy() if acc is None \
                            else np.minimum(acc, arr, out=acc)
                    elif reduction == "max":
                        acc = arr.copy() if acc is None \
                            else np.maximum(acc, arr, out=acc)
                    else:  # mean
                        if acc is None:
                            acc = arr.astype(np.float64)
                        else:
                            acc += arr

                    n_used += 1
                    break

if n_used == 0:
    print(
        f"WARNING: no frames decoded for video {video_name_no_ext} "
        "— nothing saved"
    )
    sys.exit(0)

# Finalise the reduction into a uint8 RGB image
if reduction in ("min", "max"):
    out_img = acc
else:  # mean
    out_img = np.rint(acc / n_used).astype(np.uint8)

out_path = output_dir / f"{video_name_no_ext}_{reduction}_n{n_used:04d}.png"
Image.fromarray(out_img).save(out_path)
print(f"Saved pixelwise {reduction} image ({n_used} frames) to {out_path}")
PYEOF
# -------------------------------------

echo "Computed pixelwise $REDUCTION image for video $VIDEO_NAME_NO_EXT to $OUTPUT_DIR"

# -------------------------------------------
# Move SLURM logs into the output directory
# -------------------------------------------
# Create slurm directory
LOG_DIR="$OUTPUT_DIR/logs"
mkdir -p "$LOG_DIR"

mv "slurm_array.$SLURM_ARRAY_JOB_ID-$SLURM_ARRAY_TASK_ID.$SLURMD_NODENAME.out" \
   "$LOG_DIR"
mv "slurm_array.$SLURM_ARRAY_JOB_ID-$SLURM_ARRAY_TASK_ID.$SLURMD_NODENAME.err" \
   "$LOG_DIR"

# make read-only
chmod 444 "$LOG_DIR"/slurm_array.*
