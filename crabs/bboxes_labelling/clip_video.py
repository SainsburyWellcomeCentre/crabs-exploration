"""Script to clip a video file."""

import argparse
from datetime import datetime
from pathlib import Path

import cv2


def real_time_to_frame_number(
    real_time: datetime, video_fps: float, start_real_time: datetime
) -> int:
    """Convert a real-time timestamp to the corresponding frame number.

    Parameters
    ----------
    real_time : datetime
        The real-time timestamp.
    video_fps : float
        Frames per second of the video.
    start_real_time : datetime
        The starting real-time timestamp of the video.

    Returns
    -------
    int
        The corresponding frame number in the video.

    """
    time_difference = real_time - start_real_time
    total_seconds = time_difference.total_seconds()
    return int(total_seconds * video_fps)


def create_clip(
    input_file: str, start_frame: int, end_frame: int, output_file: str
) -> None:
    """Create a video clip from the input video file.

    Parameters
    ----------
    input_file : str
        Path to the input video file.
    start_frame : int
        Starting frame number.
    end_frame : int
        Ending frame number.
    output_file : str
        Path to the output video file to be created.

    Returns
    -------
    None

    """
    cap = cv2.VideoCapture(input_file)
    video_fps = cap.get(cv2.CAP_PROP_FPS)

    fourcc = cv2.VideoWriter_fourcc(*"avc1")
    out = cv2.VideoWriter(
        output_file,
        fourcc,
        video_fps,
        (int(cap.get(3)), int(cap.get(4))),
        isColor=True,
    )

    cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret or cap.get(cv2.CAP_PROP_POS_FRAMES) > end_frame:
            break

        out.write(frame)

    cap.release()
    out.release()
    cv2.destroyAllWindows()


def argument_parser() -> argparse.Namespace:
    """Parse command-line arguments for the script.

    Returns
    -------
    argparse.Namespace
        An object containing the parsed command-line arguments.
        The attributes of this object correspond to the defined
        command-line arguments in the script.

    """
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--video_path",
        type=str,
        required=True,
        help="Location of video file.",
    )
    parser.add_argument(
        "--start_time",
        type=str,
        default="12:00:00",
        help="Start time in the format 'HH:MM:SS'.",
    )
    parser.add_argument(
        "--event_time",
        type=str,
        default="12:01:00",
        help="Event time in the format 'HH:MM:SS'.",
    )
    parser.add_argument(
        "--end_time",
        type=str,
        default="12:03:00",
        help="Time after the event in the format 'HH:MM:SS'.",
    )
    parser.add_argument(
        "--out_path",
        type=str,
        required=True,
        help="Location of video file.",
    )
    args = parser.parse_args()
    return args


if __name__ == "__main__":
    args = argument_parser()

    input_file = args.video_path
    file_name = (
        f"{Path(args.video_path).parent.stem}_"
        f"{Path(args.video_path).stem}_"
    )

    start_real_time = datetime.strptime(args.start_time, "%H:%M:%S")
    event_time = datetime.strptime(args.event_time, "%H:%M:%S")
    after_event_time = datetime.strptime(args.end_time, "%H:%M:%S")

    # Convert event times to frame numbers
    cap = cv2.VideoCapture(args.video_path)
    video_fps = cap.get(cv2.CAP_PROP_FPS)
    start_frame = real_time_to_frame_number(
        start_real_time, video_fps, start_real_time
    )
    event_frame = real_time_to_frame_number(
        event_time, video_fps, start_real_time
    )
    after_event_frame = real_time_to_frame_number(
        after_event_time, video_fps, start_real_time
    )
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    cap.release()

    # Create pre-event clip
    pre_event_clip = f"{args.out_path}/{file_name}_pre_event.mp4"
    create_clip(args.video_path, start_frame, event_frame - 1, pre_event_clip)

    # Create event clip
    event_clip = f"{args.out_path}/{file_name}_event.mp4"
    create_clip(
        args.video_path, event_frame, after_event_frame - 1, event_clip
    )

    # Create post-event clip
    post_event_clip = f"{args.out_path}/{file_name}_post_event.mp4"
    create_clip(
        args.video_path, after_event_frame, total_frames - 1, post_event_clip
    )

    print("Clips created successfully!")
