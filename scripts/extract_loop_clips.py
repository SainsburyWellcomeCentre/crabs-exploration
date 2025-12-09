import pooch
from pooch import HTTPDownloader
import subprocess
import pandas as pd
from pathlib import Path
import os
from dotenv import load_dotenv
import argparse


# def fetch_csv_from_gin():
#     """Fetch input csv using pooch."""
#     GIN_CRABSFIELD_REPO = (
#         "https://gin.g-node.org/SainsburyWellcomeCentre/CrabsField"
#     )

#     csv_filepath = pooch.retrieve(
#         url=f"{GIN_CRABSFIELD_REPO}/raw/master/crab-loops/loop-frames-ffmpeg.csv",
#         known_hash=None,
#         fname="loop-frames-ffmpeg.csv",
#         path=Path.home() / '.cache',
#         downloader=HTTPDownloader(
#             auth=(
#                 os.getenv("GIN_USERNAME"),   # use keyring?
#                 os.getenv("GIN_TOKEN")   # use keyring?
#             )
#         )
#     )


def main(args):
    """Read input csv and extract clips per row.

    If a SLURM task ID provided: each job in the array processes one row.
    """
    # Fetch csv if it does not exist
    csv_filepath = args.csv_filepath
    # if not Path(args.csv_filepath).exists():
    #     print("Input csv file not found, fetching from GIN...")
    #     csv_filepath = fetch_csv_from_gin()

    # Read csv as dataframe
    df = pd.read_csv(csv_filepath)

    # If SLURM task ID provided: process one row
    # (Note: slurm_array_task_id=0 is falsy, so we need is not None)
    if args.slurm_array_task_id is not None:
        row = df.iloc[args.slurm_array_task_id]

        print(f"Processing clip {args.slurm_array_task_id}/{len(df)}: {row['loop_clip_name']}")
        extract_clip_and_verify_count(row, args)

    # If no array_task_id provided, process all rows (for local testing) 
    else:
        print(f"Processing all {len(df)} rows")
        for idx, row in df.iterrows():
            extract_clip_and_verify_count(row, args)


def extract_clip_and_verify_count(row, args):
    # extract clip for input row
    extract_single_clip(row, args.input_dir, args.output_dir)

    # verify frame count
    if args.verify_frames:
        actual_frames, expected_frames, matches = verify_frame_count(
            Path(args.output_dir) / row['loop_clip_name'],
            row['loop_END_frame_ffmpeg']-row['loop_START_frame_ffmpeg'] + 1, # both included
        )

def extract_single_clip(row, input_dir, output_dir):
    """Extract clip using data in row."""
    # Get paths
    input_video_path = Path(input_dir) / row['video_parent_directory'] / row['video_name']
    output_video_path = Path(output_dir) / row['loop_clip_name']

    # If input video path does not exist, check if it
    # does with switched-case extension
    if not input_video_path.exists():
        input_video_path = switch_case_in_video_extension(input_video_path)

    # Prepare ffmpeg command
    ffmpeg_command = [
        "ffmpeg", 
        "-i", str(input_video_path),
        "-ss", str(row['loop_START_seconds_ffmpeg']),
        "-to", str(row['loop_END_seconds_ffmpeg']),
        "-c:v", "libx264", 
        "-pix_fmt", "yuv420p",
        "-preset", "superfast", 
        "-crf", "15",
        "-fps_mode", "passthrough",  # to preserve frame count
        str(output_video_path)
    ]

    # print command to logs
    print(' '.join(ffmpeg_command))

    # run command
    subprocess.run(ffmpeg_command, check=True)


def switch_case_in_video_extension(file_path):
    """
    Check if path with switched extension exists.
    
    Return alternative file path if it exists, otherwise throw an error.
    """
    extension = file_path.suffix
    if extension.islower():
        alternative_path =  file_path.with_suffix(extension.upper())
    elif extension.isupper():
        alternative_path =  file_path.with_suffix(extension.lower())
    
    if alternative_path.exists():
        return alternative_path
    else:
        raise FileNotFoundError(
            f"File not found: neither {file_path} or {alternative_path} "
            "are valid file paths."
        )


def verify_frame_count(input_clip, expected_frame_count):
    # Prepare ffprobe command to count frames
    ffprobe_command = [
        "ffprobe",
        "-v", "error",
        "-select_streams", "v:0",
        "-count_frames",
        "-show_entries", "stream=nb_read_frames",
        "-of", "csv=p=0",  # CSV format without headers
        str(output_video_path)
    ]

    # Run ffprobe and capture output
    result = subprocess.run(
        ffprobe_command, 
        capture_output=True, 
        text=True, 
        check=True
    )
    
    # Parse output
    actual_frames = int(result.stdout.strip())
    
    # Compare
    matches = (actual_frames == expected_frames)


    return actual_frames, expected_frames, matches



if __name__ == "__main__":
    # Define argument parser
    parser = argparse.ArgumentParser(
        description="Extract video clips as defined in input CSV using ffmpeg"
    )
    parser.add_argument(
        "--csv_filepath", 
        type=str, 
        required=True, 
        help="Path to CSV file with clip definitions"
    )
    parser.add_argument(
        "--input_dir", 
        type=str, 
        required=True, 
        help="Directory containing input videos"
    )
    parser.add_argument(
        "--output_dir", 
        type=str, 
        required=True, 
        help="Directory for output clips"
    )
    parser.add_argument(
        "--slurm_array_task_id", 
        type=int, 
        default=None, 
        help="SLURM array task ID (0-indexed). If not provided, the script processes all clips."
    )
    parser.add_argument(
        "--verify_frames", 
        action="store_true",  # the value is False by default, and becomes True when the flag is provided
        help="Verify frame count of clip matches csv value "
    )
    
    args = parser.parse_args()
    main(args)
