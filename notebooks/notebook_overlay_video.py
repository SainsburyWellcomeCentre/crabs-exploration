# %%
import cv2
import matplotlib.pyplot as plt
import numpy as np
from cycler import cycler
from movement.io import load_bboxes

# %%%%%
# Enable interactive plots
# %matplotlib widget

# %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

# load corrected ground truth
groundtruth_csv_corrected = (
    "/Users/sofia/arc/project_Zoo_crabs/escape_clips/"
    "04.09.2023-04-Right_RE_test_corrected_ST_csv_SM_corrected.csv"
)

input_video = "/Users/sofia/arc/project_Zoo_crabs/escape_clips/crabs_track_output_selected_clips/04.09.2023-04-Right_RE_test/04.09.2023-04-Right_RE_test.mp4"

# %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
# Read corrected ground truth as a movement dataset
ds_gt = load_bboxes.from_via_tracks_file(
    groundtruth_csv_corrected, fps=None, use_frame_numbers_from_file=False
)
print(ds_gt)

# Print summary
print(f"{ds_gt.source_file}")
print(f"Number of frames: {ds_gt.sizes['time']}")
print(f"Number of individuals: {ds_gt.sizes['individuals']}")

# %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
# # Read predictions as a movement dataset
file_csv = (
    "/Users/sofia/arc/project_Zoo_crabs/escape_clips/"
    "crabs_track_output_selected_clips/04.09.2023-04-Right_RE_test/predicted_tracks.csv"
)

ds_pred = load_bboxes.from_via_tracks_file(
    file_csv, fps=None, use_frame_numbers_from_file=False
)
print(ds_pred)

# Print summary
print(f"{ds_pred.source_file}")
print(f"Number of frames: {ds_pred.sizes['time']}")
print(f"Number of individuals: {ds_pred.sizes['individuals']}")


# %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
# Viz utils


def plot_trajectories(
    ds,
    list_individuals_idcs=None,
    frame_bbox=None,
    frame_trajectory=None,
    plot_id=False,
):
    """Plot the trajectories of the selected individuals in the dataset.

    Individuals are selected by their index in the dataset.

    Frame bbox is the frame to plot the bounding box of the individuals.
    If none is specified, the first frame with non-nan x-coord is used.

    Frame trajectory is the frame up to which plot the trajectory of the individuals.
    If none is specified, all frames are plotted.
    """
    fig, ax = plt.subplots(1, 1, figsize=(10, 10))

    # add color cycler to axes
    plt.rcParams["axes.prop_cycle"] = cycler(
        color=plt.get_cmap("tab10").colors
    )
    color_cycle = plt.rcParams["axes.prop_cycle"].by_key()["color"]

    # plot all if not specified
    if list_individuals_idcs is None:
        list_individuals_idcs = range(ds.sizes["individuals"])

    # plot trajectory per individual
    list_artists = []
    for ind_idx in list_individuals_idcs:
        # if frame trajectory is passed, plot only until that frame
        start_frame = ds.time[~ds.position.isnull()[:, ind_idx, 0]][0].item()
        time_slice = slice(start_frame, None)
        if frame_trajectory:
            if frame_trajectory < start_frame:
                continue  # skip if frame_trajectory is before start_frame for this individual
            time_slice = slice(
                start_frame, frame_trajectory
            )  # current frame has bbox

        # plot trajectory
        ax.scatter(
            x=ds.position[
                time_slice, ind_idx, 0
            ],  # is the origin for pytorch in the centre of the top left pixel too?
            y=ds.position[time_slice, ind_idx, 1],
            s=1,
            color=color_cycle[ind_idx % len(color_cycle)],
        )

        # add bbox at selected frame
        # if not frame specified for it: find first with non-nan x-coord
        frame_bbox_ind = frame_bbox
        if not frame_bbox_ind:
            frame_bbox_ind = start_frame
        if frame_bbox_ind < start_frame:
            continue
        top_left = (
            ds.position[frame_bbox_ind, ind_idx, :]
            - ds.shape[frame_bbox_ind, ind_idx, :] / 2
        )
        bbox = plt.Rectangle(
            xy=tuple(top_left),  # corner closer to the origin, here top left!
            width=ds.shape[frame_bbox_ind, ind_idx, 0],
            height=ds.shape[frame_bbox_ind, ind_idx, 1],
            edgecolor=color_cycle[ind_idx % len(color_cycle)],
            facecolor="none",  # Transparent fill
            linewidth=1.5,
        )
        ax.add_patch(bbox)

        # add individual ID as text at bottom right corner of bbox
        if plot_id:
            bottom_right = (
                ds.position[frame_bbox_ind, ind_idx, :]
                + ds.shape[frame_bbox_ind, ind_idx, :] / 2
            )
            ax.text(
                x=bottom_right.data[0],
                y=bottom_right.data[1],
                s=str(ind_idx),
                horizontalalignment="left",
                color=color_cycle[ind_idx % len(color_cycle)],
                fontsize=14,
            )

        # append
        # list_artists.append(sc)
        # , bbox, t))

    ax.set_aspect("equal")
    # ax.set_ylim(0, height)
    # ax.set_xlim(0, width)
    ax.invert_yaxis()  # OJO! needed if I use text
    ax.set_xlabel("x (pixels)")
    ax.set_ylabel("y (pixels)")
    ax.set_title("Ground truth")

    return fig, ax, list_artists


def plot_trajectories_on_frame(
    ds,
    frame,
    list_individuals_idcs=None,
    frame_bbox=None,
    frame_trajectory=None,
    plot_id=False,
):
    # plot trajectories
    fig, ax, list_artists = plot_trajectories(
        ds=ds,
        list_individuals_idcs=list_individuals_idcs,
        frame_bbox=frame_bbox,
        frame_trajectory=frame_trajectory,
        plot_id=plot_id,
    )

    # plot image
    im = ax.imshow(frame)
    list_artists.append(im)

    return fig, ax, list_artists


def plot_trajectories_on_video(
    ds,
    input_video,
    list_individuals_idcs=None,
    list_frame_idcs=None,
    save_frames=False,
    plot_id=False,
):
    # Open the video file
    cap = cv2.VideoCapture(input_video)

    # Get the video cap properties
    # fps = int(cap.get(cv2.CAP_PROP_FPS))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    n_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    # Get list of frames
    if not list_frame_idcs:
        list_frame_idcs = range(n_frames)

    # Plot trajectories on each frame
    # (one figure per frame)
    for frame_idx in list_frame_idcs:
        # read frame
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
        ret, frame = cap.read()

        # plot
        if ret:
            fig, ax, _ = plot_trajectories_on_frame(
                ds=ds,
                frame=frame,  # background frame
                list_individuals_idcs=list_individuals_idcs,
                frame_bbox=frame_idx,
                frame_trajectory=frame_idx,
                plot_id=plot_id,
            )

            ax.set_title(f"frame {frame_idx}")

            # format
            ax.set_aspect("equal")
            ax.set_ylim(0, height)
            ax.set_xlim(0, width)
            ax.invert_yaxis()

            # save figure
            if save_frames:
                plt.savefig(f"{frame_idx}.png")

        else:
            print(f"Error: Could not read frame {frame_idx}")

    # Release the video capture object
    cap.release()

    return fig, ax


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

    # bboxes
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
            cv2.imshow(f"Ground truth - frame {frame_idx}", frame)

            # plot bbox of each individual
            for ind_idx in list_individuals_idcs:
                centre = ds.position[frame_idx, ind_idx, :]

                # if position is nan, skip
                if centre.isnull().any().item():
                    continue

                # else plot bbox
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

                # add past trajectory in green -- drawMarker?
                past_trajectory = ds.position[:frame_idx, ind_idx, :]
                for t in past_trajectory:
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

                # add future trajectory in red -- drawMarker?
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

            # Write the frame  to the output video
            out.write(frame)

    # Release the video capture and writer objects
    cap.release()
    out.release()

    # Close all OpenCV windows (if any)
    cv2.destroyAllWindows()

    print("Video saved at", output_video_path)


# %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
# Plot labelled individuals per frame
assert (
    np.isnan(ds_gt.position.data[:, :, 0])
    == np.isnan(ds_gt.position.data[:, :, 1])
).all()

fig, ax = plt.subplots(1, 1)

ax.matshow(np.isnan(ds_gt.position.data[:, :, 0]).T, aspect="auto")
ax.tick_params(labelright=True)
ax.set_title("Ground truth")
ax.set_xlabel("time (frames)")
ax.set_ylabel("individual")

# %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
# Plot all GT and predicted trajectories
fig, ax, _ = plot_trajectories(ds=ds_gt)
ax.set_title("Ground truth")


fig, ax, _ = plot_trajectories(ds=ds_pred)
ax.set_title("Predicted")

fig, ax, _ = plot_trajectories(ds=ds_pred, plot_id=False)
ax.set_title("Predicted")
# %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
# Plot all GT and predicted trajectories on frame 0

fig, ax = plot_trajectories_on_video(
    ds=ds_gt, input_video=input_video, list_frame_idcs=[0]
)
ax.set_title("Ground truth - frame 0")

fig, ax = plot_trajectories_on_video(
    ds=ds_pred, input_video=input_video, list_frame_idcs=[0], plot_id=False
)
ax.set_title("Prediction - frame 0")
# %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
# Plot one individual's GT trajectories on frame 50
plt.ioff()
fig, ax = plot_trajectories_on_video(
    ds=ds_gt,
    input_video=input_video,
    list_frame_idcs=[50],
    list_individuals_idcs=[34],
)


# %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
# Plot one individual's GT trajectories all frames and save
plt.ioff()
fig, ax = plot_trajectories_on_video(
    ds=ds_gt,
    input_video=input_video,
    list_individuals_idcs=[32],
    save_frames=True,
)
