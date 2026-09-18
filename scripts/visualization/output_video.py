"""Script to create a video with tracked bounding boxes."""

import os
from pathlib import Path

import cv2
import matplotlib.pyplot as plt
import numpy as np
from movement.io import load_bboxes
from tqdm import tqdm


def get_distinct_colors():
    """Generate 100+ distinct colors from matplotlib qualitative colormaps."""
    # Use colormaps that don't overlap with each other
    colormaps = [
        "tab20",
        "tab20b",
        "tab20c",
        "Set1",
        "Set2",
        "Set3",
        "Dark2",
        "Accent",
    ]
    colors = []
    for cmap_name in colormaps:
        cmap = plt.get_cmap(cmap_name)
        n_colors = cmap.N
        for i in range(n_colors):
            rgba = cmap(i)
            # Convert from RGB (0-1) to BGR (0-255) for OpenCV
            bgr = (int(rgba[2] * 255), int(rgba[1] * 255), int(rgba[0] * 255))
            colors.append(bgr)
    return colors


def reindex_to_video_clip_frames(ds, video_path):
    """Pad the dataset's time coordinate to span every frame of the clip.

    The dataset is expected to hold the frame numbers 0-based indices over
    the clip, as written in the VIA tracks file (i.e. loaded with
    `use_frame_numbers_from_file=True` and `fps=None`).
    """
    # Get n frames in video clip
    cap = cv2.VideoCapture(str(video_path))
    n_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    cap.release()

    # Check the frame indices in the file are within the video range
    frames_0idx = ds.time.values
    if frames_0idx.min() < 0 or frames_0idx.max() >= n_frames:
        raise ValueError(
            f"{Path(video_path).name}: the frame numbers in the tracks file "
            f"({frames_0idx.min()}-{frames_0idx.max()}) fall outside the "
            f"video's frame range 0-{n_frames - 1}. Are the tracks from this "
            "clip?"
        )

    # Print number of frames with no detections
    n_frames_wo_boxes = n_frames - ds.sizes["time"]
    if n_frames_wo_boxes:
        print(
            f"{Path(video_path).name}: {n_frames_wo_boxes} of {n_frames} "
            "frames have no tracked boxes; padding them with NaNs."
        )

    # Reindex dataset to number of frames in video
    # (reindexing fills in any gaps with nans, but silently drops
    # frames out of the specificied range; hence the range check earlier)
    return ds.reindex(time=np.arange(n_frames))


def create_opencv_video(
    ds,
    input_video,
    output_video_path,
    list_individuals_idcs=None,
    list_frame_idcs=None,
):
    """Create a video with bounding boxes around the selected individuals."""
    # Open the video file
    cap = cv2.VideoCapture(input_video)

    # Get the video cap properties
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    n_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    # Prepare video writer
    # Define the codec and create VideoWriter object to save the output video
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")  # Codec for mp4
    out = cv2.VideoWriter(output_video_path, fourcc, fps, (width, height))

    # Get distinct colors for each individual
    colors = get_distinct_colors()

    # Get list of frames
    if not list_frame_idcs:
        list_frame_idcs = range(n_frames)

    # Plot trajectories per frame
    for frame_idx in tqdm(list_frame_idcs):
        # read frame
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
        ret, frame = cap.read()

        if ret:
            # plot frame
            # cv2.imshow(f"frame {frame_idx}", frame)

            # plot bbox of each individual
            for ind_idx in list_individuals_idcs:
                # Get color for this individual
                color_indiv = colors[ind_idx % len(colors)]

                # add title with frame number
                cv2.putText(
                    frame,
                    f"Frame {frame_idx}",
                    (
                        int(0.8 * width),
                        int(width / 30),
                    ),  # location of text bottom left
                    cv2.FONT_HERSHEY_SIMPLEX,
                    3,  # fontsize
                    (255, 0, 0),  # BGR
                    6,  # thickness
                    cv2.LINE_AA,
                )

                # if position is not nan, plot bbox with ID
                centre = ds.position[frame_idx, :, ind_idx]
                if not centre.isnull().any().item():
                    top_left = centre - ds.shape[frame_idx, :, ind_idx] / 2
                    bottom_right = centre + ds.shape[frame_idx, :, ind_idx] / 2
                    cv2.rectangle(
                        frame,
                        tuple(int(x) for x in top_left),
                        tuple(int(x) for x in bottom_right),
                        color_indiv,
                        3,  # rectangle_thickness
                    )

                # add past trajectory with individual's color,
                # including current frame
                past_trajectory = ds.position[: frame_idx + 1, :, ind_idx]
                for t in past_trajectory:
                    # skip if nan
                    if t.isnull().any().item():
                        continue
                    x, y = t.data
                    cv2.circle(
                        frame,
                        (int(x), int(y)),  # centre
                        3,  # radius
                        color_indiv,
                        -1,
                    )

                # add future trajectory with faded individual's color
                fut_trajectory = ds.position[frame_idx + 1 :, :, ind_idx]
                # Create faded color (50% brightness)
                # faded_color = tuple(int(c * 0.5) for c in color_indiv)
                for t in fut_trajectory:
                    if t.isnull().any().item():
                        continue
                    x, y = t.data
                    cv2.circle(
                        frame,
                        (int(x), int(y)),  # centre
                        3,  # radius
                        (255, 255, 255),  # white #faded_color,
                        -1,
                    )

            # Write the frame to the output video
            out.write(frame)

    # Release the video capture and writer objects
    cap.release()
    out.release()

    # Close all OpenCV windows (if any)
    cv2.destroyAllWindows()

    print("Video saved at", output_video_path)


if __name__ == "__main__":
    # input/output locations
    input_data_dir = (
        "/Users/sofia/arc/project_Zoo_crabs/videos_presentation_Tiago/"
        "tracking_output_above_10th_percentile_slurm_1249954"
    )
    output_data_dir = Path(
        "/Users/sofia/arc/project_Zoo_crabs/videos_presentation_Tiago/output"
    )

    list_escape_clips = sorted(
        [
            Path(file).stem
            for file in os.listdir(input_data_dir)
            if file.endswith(".mp4")
        ]
    )
    for escape_clip_name in list_escape_clips[-1:]:
        input_clip = Path(input_data_dir) / f"{escape_clip_name}.mp4"
        pred_csv = Path(input_data_dir) / f"{escape_clip_name}_tracks.csv"

        # Read predictions as a movement dataset.
        # We request the frame numbers as defined in the file
        # (only frames with detections are included in the file)
        ds_pred = load_bboxes.from_via_tracks_file(
            pred_csv,
            fps=None,
            use_frame_numbers_from_file=True,
        )

        # Pad the time coordinate to span the full video, so that time
        # coordinate i of the dataset is frame index (0-based) i of the clip
        ds_pred = reindex_to_video_clip_frames(ds_pred, input_clip)

        list_individuals_idcs = list(range(len(ds_pred.individuals)))

        # Create prediction video
        output_video_path = (
            output_data_dir / f"{escape_clip_name}_bboxes_tracks.mp4"
        )
        create_opencv_video(
            ds=ds_pred,
            input_video=input_clip,
            output_video_path=output_video_path,
            list_individuals_idcs=list_individuals_idcs,
        )
