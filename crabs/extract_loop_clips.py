import subprocess
import pandas as pd
from pathlib import Path
import os
import argparse
import sys

def main(args: argparse.Namespace) -> None:
    """Read input csv and extract loop clips defined per row.

    If a SLURM task ID is provided: each job in the array processes one row.

    Parameters
    ----------
    args : argparse.Namespace
        Arguments parsed from command line. Should contain:
        - csv_filepath: path to the input csv file.
        - input_dir: path to the input directory containing the input videos.
        - output_dir: path to the output directory for the extracted loop clips.
        - slurm_array_task_id: SLURM array task ID (0-indexed). If not provided, the script processes all clips.
        - verify_frames: whether to verify frame count of the extracted clips matches the value in the csv file.
    """
    # Read csv as dataframe
    df = pd.read_csv(args.csv_filepath)

    # If SLURM task ID provided: process one row
    # (Note: slurm_array_task_id=0 is falsy, so we need "is not None")
    if args.slurm_array_task_id is not None:
        row = df.iloc[args.slurm_array_task_id]

        print(f"Processing clip {args.slurm_array_task_id}/{len(df)}: {row['loop_clip_name']}")
        extract_clip_and_verify_count(row, args)

    # If no slurm_array_task_id is provided, process all rows (for local testing) 
    else:
        print(f"Processing all {len(df)} rows in csv file: {args.csv_filepath}")
        for idx, row in df.iterrows():
            extract_clip_and_verify_count(row, args)


def extract_clip_and_verify_count(row: pd.Series, args: argparse.Namespace) -> None:
    """Extract clip for input row and verify frame count.
    
    Parameters
    ----------
    row : pandas.Series
        Row in csv file containing clip definition.
    args : argparse.Namespace
        Arguments parsed from command line. Should contain:
        - input_dir: path to the input directory containing the input videos.
        - output_dir: path to the output directory for the extracted loop clips.
        - verify_frames: whether to verify frame count of the extracted clips matches the value in the csv file.

    Raises
    ------
    FileNotFoundError: if the input video path does not exist.
    subprocess.CalledProcessError: if the ffmpeg command fails.
    """
    # extract clip for input row
    extract_single_clip(row, args.input_dir, args.output_dir)

    # verify frame count
    if args.verify_frames:
        # Compute expected frame count
        expected_frame_count = row['loop_END_frame_ffmpeg']-row['loop_START_frame_ffmpeg'] + 1 # both included

        # Verify frame count matches csv expected value
        frame_count_ok, actual_frame_count = verify_frame_count(
            Path(args.output_dir) / row['loop_clip_name'],
            expected_frame_count,
        )
        print(f"Expected frame count: {expected_frame_count}")
        print(f"Actual frame count: {actual_frame_count}")
        if frame_count_ok:
            print(f"Frame count OK for clip {row['loop_clip_name']}")
        else:
            print(f"Frame count MISMATCH for clip {row['loop_clip_name']}")
           

def extract_single_clip(row: pd.Series, input_dir: str | Path, output_dir: str | Path) -> None:
    """Extract clip using data in row.
    
    Parameters
    ----------
    row : pandas.Series
        Row in the input csv file containing the loop clip definition. It should contain the following columns:
        - video_parent_directory: name of the parent directory of the input video. It is usually a subdirectory 
          of the input directory that refers to the date of the experiments (e.g. `09.08.2023-Day2`).
        - video_name: name of the input video.
        - loop_clip_name: name of the output clip. It includes the file extension (e.g. `.mp4`).
        - loop_START_seconds_ffmpeg: timestamp in seconds for the first frame to include in the output clip.
        - loop_END_seconds_ffmpeg: timestamp in seconds for the last frame to include in the output clip.
    input_dir : str | Path
        Path to the input directory containing the input videos.
    output_dir : str | Path
        Path to the output directory for the extracted loop clips.

    Raises
    ------
    FileNotFoundError: if the input video path does not exist.
    subprocess.CalledProcessError: if the ffmpeg command fails.

    Notes
    -----
    The ffmpeg command assumes frame numbering starts at 1 (1-based indexing). If the input video path does not exist, 
    we additionally check if it exists with the switched-case extension (e.g. `.MOV` instead of `.mov`).

    """
    # Get paths to input and output videos
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


def verify_frame_count(input_clip: str | Path, expected_frame_count: int) -> tuple[bool, int]:
    """Verify that the frame count of input clip matches the expected value.
    
    Parameters
    ----------
    input_clip : str
        Path to the input clip.
    expected_frame_count : int
        Expected frame count.

    Returns
    -------
    tuple: (boolean, int)
        - boolean: True if the frame count matches the expected value, False otherwise.
        - int: the frame count returned by ffprobe.

    Raises
    ------
    subprocess.CalledProcessError: if the ffprobe command fails.
    """
    # Prepare ffprobe command to count frames
    ffprobe_command = [
        "ffprobe",
        "-v", "error",
        "-select_streams", "v:0",
        "-count_frames",
        "-show_entries", "stream=nb_read_frames",
        "-of", "csv=p=0",  # output as .csv without headers
        str(input_clip)
    ]

    # Run ffprobe and capture output
    result = subprocess.run(
        ffprobe_command, 
        capture_output=True, 
        text=True, 
        check=True
    )
    
    # Parse output
    actual_frame_count = int(result.stdout.strip())
    
    # Return boolean and actual frame counts
    return (
        actual_frame_count == expected_frame_count,
        actual_frame_count
    )


def switch_case_in_video_extension(file_path: str | Path) -> Path:
    """
    Check if the input file path with the switched-case extension exists.
    
    Parameters
    ----------
    file_path : str | Path
        Path to the input file.

    Returns
    -------
    Path
        The input file path with the switched-case extension if 
        it exists.

    Raises
    ------
    FileNotFoundError
        If the input file path with the switched-case extension does not exist.
    """
    # Compute alternative filepath with switched-case extension
    extension = file_path.suffix
    if extension.islower():
        alternative_path =  file_path.with_suffix(extension.upper())
    elif extension.isupper():
        alternative_path =  file_path.with_suffix(extension.lower())
    
    # If alternative path exists, return it
    if alternative_path.exists():
        return alternative_path

    # If it does not exist, throw an error
    else:
        raise FileNotFoundError(
            f"File not found: neither {file_path} or {alternative_path} "
            "are valid file paths."
        )


def parse_args(args: list[str]) -> argparse.Namespace:
    """Parse command line arguments for the loop clips extraction.
    
    Parameters
    ----------
    args : list[str]
        Command line arguments.

    Returns
    -------
    argparse.Namespace
        Arguments parsed from command line.
    """
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
    
    return parser.parse_args(args)


def app_wrapper():
    """Wrap function for extracting loop clips."""
    args = parse_args(sys.argv[1:])
    main(args)

if __name__ == "__main__":
    app_wrapper()