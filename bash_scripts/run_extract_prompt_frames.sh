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

# -----------------------------------------------------------
# Extract all frames for this clip in a single ffmpeg pass
# -----------------------------------------------------------

# Build compound select expression
# eg: for frames 100, 234 and 512 --> eq(n,100)+eq(n,234)+eq(n,512)
# (n is ffmpeg's built-in variable for the zero-based index of the current video frame being decoded.)
#
# IMPORTANT NOTE: the order inside the select expression doesn't matter for ffmpeg's output,
# because eq(n,4)+eq(n,100)+... is a logical "OR" and ffmpeg emits matching frames in
# decode order (that is, ascending order)
SELECT_EXPR=$(printf "eq(n,%d)+" "${FRAME_IDCS[@]}" | sed 's/+$//')


# Run ffmpeg command to extract frames in decode order
# (we assume input video is reencoded such that frames are reliably
# seekable)
# ffmpeg decodes from 0 to the largest frame
# -------------------------------------
# TODO: change to pyAV
# Create temporary directory
TMP_DIR=$(mktemp -d)

ffmpeg -loglevel error \
    -i "$CLIP_PATH" \
    -vf "select='${SELECT_EXPR}'" \
    -vsync vfr \  # prevent ffmpeg from producing a video with same fps as input
    -q:v 2 \
    "${TMP_DIR}/frame_%08d.png"

# Since frames are extracted
# ffmpeg names its outputs sequentially: frame_00000001.png, frame_00000002.png, etc.,
# regardless of which frame indices were extracted. This loop remaps those sequential
# names back to the actual frame indices.
for i in "${!FRAME_IDCS[@]}"; do
    src="${TMP_DIR}/$(printf 'frame_%08d.png' $((i + 1)))"
    dst="${OUTPUT_DIR_JOB}/${CLIP_NAME_NO_EXT}_frame_$(printf '%08d' "${FRAME_IDCS[$i]}").png"
    mv "$src" "$dst"
done

rm -rf "$TMP_DIR"
# -------------------------------------

echo "Extracted ${#FRAME_IDCS[@]} frames from $CLIP_NAME to $OUTPUT_DIR_JOB"


# ffmpeg's select filter seeks to the nearest preceding keyframe, then decodes
# forward to each target frame — should be functionally equivalent to the
# PyAV approach:
#
#   with av.open(clip_path) as container:
#       stream = container.streams.video[0]
#       fps = stream.codec_context.framerate  # Fraction
#       container.seek(int(target_idx / float(fps) * 1e6))
#       target_pts = None
#       for frame in container.decode(stream):
#           if target_pts is None:
#               target_pts = int(Fraction(target_idx) / fps / frame.time_base)
#           if frame.pts is None:
#               raise ValueError("Frame has no PTS")
#           if frame.pts >= target_pts:
#               img = frame.to_ndarray(format="rgb24")
#               break


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
