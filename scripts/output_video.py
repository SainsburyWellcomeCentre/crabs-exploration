from pathlib import Path

import cv2


def create_opencv_video(
    ds,
    input_video,
    output_video_path,
    list_individuals_idcs=None,
    list_frame_idcs=None,
):
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

    # bboxes format
    rectangle_color = (0, 255, 0)  # Green color in BGR format

    # Get list of frames
    if not list_frame_idcs:
        list_frame_idcs = range(n_frames)

    # Plot trajectories per frame
    for frame_idx in list_frame_idcs:
        # read frame
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
        ret, frame = cap.read()

        if ret:
            # plot frame
            cv2.imshow(f"frame {frame_idx}", frame)

            # plot bbox of each individual
            for ind_idx in list_individuals_idcs:
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
                centre = ds.position[frame_idx, ind_idx, :]
                if not centre.isnull().any().item():
                    top_left = centre - ds.shape[frame_idx, ind_idx, :] / 2
                    bottom_right = centre + ds.shape[frame_idx, ind_idx, :] / 2
                    cv2.rectangle(
                        frame,
                        tuple(int(x) for x in top_left),
                        tuple(int(x) for x in bottom_right),
                        rectangle_color,
                        3,  # rectangle_thickness
                    )

                    # add ID
                    cv2.putText(
                        frame,
                        str(ind_idx),
                        tuple(
                            int(x) for x in bottom_right
                        ),  # location of text bottom left
                        cv2.FONT_HERSHEY_SIMPLEX,
                        2,  # fontsize
                        rectangle_color,
                        6,  # thickness
                        cv2.LINE_AA,
                    )

                # add past trajectory in green -- drawMarker?
                past_trajectory = ds.position[:frame_idx, ind_idx, :]
                for t in past_trajectory:
                    # skip if nan
                    if t.isnull().any().item():
                        continue
                    x, y = t.data
                    cv2.circle(
                        frame,
                        (int(x), int(y)),  # centre
                        3,  # radius
                        (0, 255, 0),  # BGR
                        -1,
                    )

                # add future trajectory in grey -- drawMarker?
                fut_trajectory = ds.position[frame_idx:, ind_idx, :]
                for t in fut_trajectory:
                    if t.isnull().any().item():
                        continue
                    x, y = t.data
                    cv2.circle(
                        frame,
                        (int(x), int(y)),  # centre
                        3,  # radius
                        (255, 255, 255),  # BGR
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
    from movement.io import load_bboxes

    # load input data
    groundtruth_csv_corrected = (
        "/Users/sofia/arc/project_Zoo_crabs/escape_clips/"
        "04.09.2023-04-Right_RE_test_corrected_ST_csv_SM_corrected.csv"
    )

    pred_csv = (
        "/Users/sofia/arc/project_Zoo_crabs/escape_clips/"
        "crabs_track_output_selected_clips/04.09.2023-04-Right_RE_test/predicted_tracks.csv"
    )

    input_video = (
        "/Users/sofia/arc/project_Zoo_crabs/escape_clips/crabs_track_output_selected_clips/"
        "04.09.2023-04-Right_RE_test/04.09.2023-04-Right_RE_test.mp4"
    )

    # Read corrected ground truth as a movement dataset
    ds_gt = load_bboxes.from_via_tracks_file(
        groundtruth_csv_corrected, fps=None, use_frame_numbers_from_file=False
    )

    # Read predictions as a movement dataset
    ds_pred = load_bboxes.from_via_tracks_file(
        pred_csv, fps=None, use_frame_numbers_from_file=False
    )

    # # Create ground truth video
    # list_individuals_idcs = [32]
    # for id in list_individuals_idcs:
    #     output_video_path = str(Path(__file__).parent / f"gt_id_{id}.mp4")
    #     # ground truth video
    #     create_opencv_video(
    #         ds=ds_gt,
    #         input_video=input_video,
    #         output_video_path=output_video_path,
    #         list_individuals_idcs=list_individuals_idcs,
    #     )

    # Create prediction video
    list_individuals_idcs = [22]
    output_video_path = str(
        Path(__file__).parent
        / f"pred_id_{'_'.join([str(el) for el in list_individuals_idcs])}.mp4"
    )
    create_opencv_video(
        ds=ds_pred,
        input_video=input_video,
        output_video_path=output_video_path,
        list_individuals_idcs=list_individuals_idcs,
    )
