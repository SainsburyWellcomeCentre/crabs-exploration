import pooch
from pooch import HTTPDownloader
import subprocess
import pandas as pd
from pathlib import Path
import os
from dotenv import load_dotenv
import argparse

# GIN_CRABSFIELD_REPO = (
#     "https://gin.g-node.org/SainsburyWellcomeCentre/CrabsField"
# )
# Fetch loops csv using pooch
# csv_filepath = pooch.retrieve(
#     url=f"{GIN_CRABSFIELD_REPO}/raw/master/crab-loops/loop-frames-ffmpeg.csv",
#     known_hash=None,
#     fname="loop-frames-ffmpeg.csv",
#     path=Path.home() / '.cache',
#     downloader=HTTPDownloader(auth=(os.getenv("GIN_USERNAME"), os.getenv("GIN_TOKEN")))
# )


def extract_single_clip(row, input_dir, output_dir):
    # Get paths
    input_video_path = Path(input_dir) / row['video_name']
    output_video_path = Path(output_dir) / row['loop_clip_name']

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
        "-vstats", 
        "-fps_mode", "passthrough",
        str(output_video_path)
    ]

    # print to logs
    print(' '.join(ffmpeg_command))

    # run
    subprocess.run(ffmpeg_command, check=True, capture_output=True, text=True)



def main(args):
    # Read as dataframe
    # video_name,loop_clip_name,loop_START_frame_ffmpeg,loop_END_frame_ffmpeg,escape_START_frame_ffmpeg,escape_type
    df = pd.read_csv(args.csv_filepath)

    # If SLURM task ID provided: process one row
    # (Note: a SLURM task ID equal to 0 evalute to False, so we need is not None)
    if args.slurm_array_task_id is not None:
        row = df.iloc[args.slurm_array_task_id]

        print(f"Processing clip {args.slurm_array_task_id}/{len(df)}: {row['loop_clip_name']}")
        extract_single_clip(row, args.input_dir, args.output_dir)

    # If no array_task_id provided, process all rows (local testing) 
    else:
        print(f"Processing all {len(df)} rows")
        for idx, row in df.iterrows():
            extract_single_clip(row, args.input_dir, args.output_dir)


if __name__ == "__main__":
    # Define argument parser
    parser = argparse.ArgumentParser(
        description="Extract video clips from CSV specification using ffmpeg"
    )
    parser.add_argument(
        "--csv_filepath", 
        type=str, 
        required=True, 
        help="Path to CSV file with clip specifications"
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
        help="SLURM array task ID (0-indexed). If not provided, processes all clips."
    )
    
    args = parser.parse_args()
    main(args)
    
    args = parser.parse_args()
    main(args)