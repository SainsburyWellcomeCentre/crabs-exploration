"""A script to extract frames for labelling using SLEAP's algorithm.

TODO: can I make it deterministic?
TODO: check https://github.com/talmolab/sleap-io/tree/main/sleap_io
TODO: change it to copy directory structure from input? See
https://www.geeksforgeeks.org/python-copy-directory-structure-without-files/
"""

import copy
import json
import logging
import pprint
import sys
from datetime import datetime
from pathlib import Path
from typing import Optional

import cv2
import typer
from sleap import Video
from sleap.info.feature_suggestions import (
    FeatureSuggestionPipeline,
    ParallelFeaturePipeline,
)

# instantiate Typer app
app = typer.Typer(rich_markup_mode="rich")


def get_list_of_sleap_videos(  # noqa: C901
    list_video_locations,
    video_extensions_in=("mp4"),
):
    """Generate list of SLEAP videos.

    The locations in which we look for videos
    can be expressed as paths to files or
    as the parent directories of a set of videos.

    Parameters
    ----------
    list_video_locations : list[str]
        list of video locations. These may be paths to video files or
        paths to their parent directories (only one level deep is searched).

    video_extensions_in : tuple[str]
        list of video extensions to look for in the directories.
        By default, mp4 videos.

    Returns
    -------
    list_sleap_videos : list[sleap.io.video.Video]
        list of SLEAP videos

    """
    # Make list of extensions case insensitive
    list_video_extensions_in = list(video_extensions_in)
    list_video_extensions = copy.deepcopy(list_video_extensions_in)
    for ext in list_video_extensions_in:
        if ext.isupper():
            list_video_extensions.append(ext.lower())
        elif ext.islower():
            list_video_extensions.append(ext.upper())

    # Compute list of video paths
    list_video_paths = []
    for loc in list_video_locations:
        location_path = Path(loc)

        # If the path is a directory:
        # look for files with any of the relevant extensions
        # (only one level in)
        if location_path.is_dir():
            for ext in list_video_extensions:
                list_video_paths.extend(
                    location_path.glob(f"[!.]*.{ext}"),
                )  # exclude hidden files

        # If the path is a file with the relevant extension:
        # append path directly to list
        elif location_path.is_file() and (
            location_path.suffix[1:] in list_video_extensions
            # suffix includes dot
        ):
            list_video_paths.append(location_path)

    # Transform list of video paths to list of SLEAP videos
    list_sleap_videos = []
    for vid_path in list_video_paths:
        # check if opencv can open the videos
        # before adding them to the list
        cap = cv2.VideoCapture(str(vid_path))
        if cap.isOpened():
            list_sleap_videos.append(Video.from_filename(str(vid_path)))
            cap.release()
        else:
            logging.warning(
                f"Video at {vid_path!s} could not"
                " be opened by OpenCV. Skipping...",
            )

    # Print warning if list is empty
    if not list_sleap_videos:
        logging.error(
            "List of videos is empty. Please review: \n"
            f"\t input video locations:{list_video_locations}\n "
            f"\t input video extensions:{list_video_extensions})\n",
        )
        sys.exit(1)

    return list_sleap_videos


def get_map_videos_to_extracted_frames(list_sleap_videos, suggestions):
    """Compute dictionary mapping videos to frame indices for labelling.

    Parameters
    ----------
    list_sleap_videos : list[sleap.io.video.Video]
        list of SLEAP videos from which to extract frames

    suggestions : list[SuggestionFrame]
        a list of SuggestionFrame elements, describing
        the frames selected for labelling

    Returns
    -------
    map_videos_to_extracted_frames : dict
        dictionary that maps each video path to a list
        of frames indices extracted for labelling.
        The frame indices are sorted in ascending order.

    """
    map_videos_to_extracted_frames = {}
    for vid in list_sleap_videos:
        vid_str = vid.backend.filename
        map_videos_to_extracted_frames[vid_str] = sorted(
            [
                sugg.frame_idx
                for sugg in suggestions
                if sugg.video.backend.filename == vid_str
            ],
        )
    return map_videos_to_extracted_frames


def compute_suggested_sleap_frames(
    list_video_locations,
    video_extensions=("mp4"),
    initial_samples=200,
    sample_method="stride",
    scale=1.0,
    feature_type="raw",
    n_components=5,
    n_clusters=5,
    per_cluster=5,
    compute_features_per_video=True,
):
    """Compute frames for labelling using SLEAP's FeatureSuggestionPipeline.

    See https://sleap.ai/guides/gui.html#labeling-suggestions

    Parameters
    ----------
    list_video_locations : list[str]
        list of video locations. These may be paths to video files or
        paths to their parent directories (only one level deep is searched).
    video_extensions : tuple[str]
        tuple of video extensions to look for in the directories.
        Default: ("mp4")
    initial_samples : int
        initial number of frames to extract per video
        Default: 200
    sample_method : str
        method to sample initial samples.
        It can be "random" or "stride".
        Default: "stride"
    scale : float
        factor to apply to the images prior to PCA and k-means clustering
        Default: 1.0
    feature_type : str
        type of input feature.
        It can be ["raw", "brisk", "hog"].
        Default: raw
    n_components : int
        number of PCA components.
        Default: 5
    n_clusters : int
        number of k-means clusters.
        Default: 5
    per_cluster : int
        number of frames to sample per cluster.
        Default: 5
    compute_features_per_video : bool
        whether to do per-video pipeline parallelization for
        feature suggestions.
        Default: True

    Returns
    -------
    map_videos_to_extracted_frames : dict
        dictionary that maps each video path to a list
        of frames indices extracted for labelling.
        The frame indices are sorted in ascending order.

    """
    # Transform list of input videos to list of SLEAP Video instances
    list_sleap_videos = get_list_of_sleap_videos(
        list_video_locations,
        video_extensions,
    )
    logging.info("List of SLEAP videos successfully created")

    # Define the frame extraction pipeline
    pipeline = FeatureSuggestionPipeline(
        per_video=initial_samples,
        sample_method=sample_method,
        scale=scale,
        feature_type=feature_type,
        n_components=n_components,
        n_clusters=n_clusters,
        per_cluster=per_cluster,
    )
    logging.info("---------------------------")
    logging.info("Defintion of FeatureSuggestionPipeline:")
    pipeline_attrs = {
        k: getattr(pipeline, k) for k in dir(pipeline) if not k.startswith("_")
    }
    logging.info(pprint.pformat(pipeline_attrs))
    logging.info("---------------------------")

    # Run the pipeline and compute  suggested frames for labelling
    # (if compute_features_per_video=True, it is run per video)
    suggestions = ParallelFeaturePipeline.run(
        pipeline,
        list_sleap_videos,
        parallel=compute_features_per_video,
    )
    logging.info(f"Total labelling suggestions generated: {len(suggestions)}")

    # Compute dictionary that maps video paths to their frames' indices
    # suggested for labelling
    return get_map_videos_to_extracted_frames(
        list_sleap_videos,
        suggestions,
    )


def extract_frames_to_label_from_video(
    map_videos_to_extracted_frames,
    output_subdir_path,
    flag_parent_dir_subdir_in_output=False,
):
    """Extract frames for labelling from corresponding videos using OpenCV.

    The png files for each frame are named with
    the following format:
    <video_parent_dir>_<video_filename>_frame_<frame_idx>.png

    Parameters
    ----------
    map_videos_to_extracted_frames : dict
        dictionary that maps each video path to a list
        of frames indices extracted for labelling.
        The frame indices are sorted in ascending order.

    output_subdir_path : pathlib.Path
        path to output subdirectory

    flag_parent_dir_subdir_in_output : bool
        if True, a subdirectory is created under 'output_subdir_path'
        whose name matches the video's parent directory name

    Raises
    ------
    KeyError
        If a frame from a video is not correctly read by openCV

    """
    for vid_str in map_videos_to_extracted_frames:
        # Initialise video capture
        cap = cv2.VideoCapture(vid_str)

        # Check if video capture is opened correctly
        logging.info("---------------------------")
        if cap.isOpened():
            logging.info(f"Processing video {Path(vid_str)}")
        else:
            logging.info(f"Error processing {Path(vid_str)}, skipped....")
            continue

        # If required: create video output dir inside timestamped one
        if flag_parent_dir_subdir_in_output:
            video_output_dir = (
                output_subdir_path
                / Path(
                    vid_str
                ).parent.stem  # timestamp  # parent dir of input video
            )
            video_output_dir.mkdir(parents=True, exist_ok=True)
        else:
            video_output_dir = output_subdir_path

        # Go to the selected frames in the video
        for frame_idx in map_videos_to_extracted_frames[vid_str]:
            # Read frame
            # TODO: are sleap suggested frame numbers indices (i.e. 0-based)
            # or frame numbers (1-based)
            cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
            success, frame = cap.read()

            # If not read successfully: throw error
            if not success or frame is None:
                msg = f"Unable to load frame {frame_idx} from {vid_str}."
                raise KeyError(msg)

            # If read successfully: save to file
            # file naming format: videoname_frame_XXX.png
            else:
                file_path = video_output_dir / Path(
                    f"{Path(vid_str).stem}_" f"frame_{frame_idx:08d}.png",
                )
                img_saved = cv2.imwrite(str(file_path), frame)
                if img_saved:
                    logging.info(f"frame {frame_idx} saved at {file_path}")
                else:
                    logging.info(
                        f"ERROR saving {Path(vid_str).stem}, frame {frame_idx}"
                        "...skipping",
                    )
                    continue

        # close video capture
        cap.release()


@app.command()
def compute_and_extract_frames_to_label(
    list_video_locations: list[str],
    output_path: str = ".",
    output_subdir: Optional[str] = None,
    video_extensions: tuple[str] = ("mp4",),
    initial_samples: int = 200,
    sample_method: str = "stride",  # choices=["random", "stride"],
    scale: float = 1.0,
    feature_type: str = "raw",  # choices=["raw", "brisk", "hog"],
    n_components: int = 5,
    n_clusters: int = 5,
    per_cluster: int = 5,
    compute_features_per_video: bool = True,
):
    """Compute frames to label and extract them as png files.

    We use SLEAP's image feature method to select
    the frames for labelling and export them as png
    files in the desired directory.

    We also output to the same location the list of
    frame indices selected per video as a json file.

    Parameters
    ----------
    list_video_locations : list
        list of paths to directories with videos, or to specific video files.
    output_path : str, optional
        path to directory in which to store extracted frames, by default the
        current directory.
    output_subdir : str, optional
        name of output subdirectory in which to put extracted frames,
        by default the timestamp in the format YYYMMDD_HHMMSS.
    video_extensions : tuple, optional
        extensions to search for when looking for video files,
        by default ("mp4")
    initial_samples : int, optional
        initial number of frames to extract per video, by default 200
    sample_method : str, optional
        method to sample initial frames, a choice between "random" or "stride,
        by default "stride"
    scale : float, optional
        factor to apply to the images prior to PCA and k-means clustering,
        by default 1.0
    feature_type : str, optional
        type of input feature, a choice between "raw", "brisk" or "hog",
        by default "raw"
    n_components : int, optional
        number of PCA components to compute, by default 5
    n_clusters : int, optional
        number of k-means clusters to compute, by default 5
    per_cluster : int, optional
        number of frames to sample per cluster, by default 5
    compute_features_per_video : bool, optional
        whether to compute the (PCA?) features per video, or across all videos,
        by default True

    """
    # Compute list of suggested frames using SLEAP
    map_videos_to_extracted_frames = compute_suggested_sleap_frames(
        list_video_locations,
        video_extensions,
        initial_samples,
        sample_method,
        scale,
        feature_type,
        n_components,
        n_clusters,
        per_cluster,
        compute_features_per_video,
    )

    # Create target subdirectory inside the output folder, if it doesn't exist.
    # If no output subdirectory name is provided, create one whose name
    # is the current timestamp in the format YYYYMMDD_HHMMSS
    if not output_subdir:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_subdir_path = Path(output_path) / f"{timestamp}"
    else:
        output_subdir_path = Path(output_path) / output_subdir
    output_subdir_path.mkdir(parents=True, exist_ok=True)

    # Save the set of videos and corresponding extracted frames' indices
    # as json file
    json_output_file = output_subdir_path / "extracted_frames.json"
    # if json file exists: append
    if json_output_file.is_file():
        with open(json_output_file) as js:
            map_pre = json.load(js)
            map_pre.update(map_videos_to_extracted_frames)
        with open(json_output_file, "w") as js:
            json.dump(
                map_pre,
                js,
                sort_keys=True,
                indent=4,
            )
        logging.info(
            "Existing json file with "
            f"extracted frames updated at {json_output_file}",
        )
    # else: start a new file
    else:
        with open(json_output_file, "w") as js:
            json.dump(
                map_videos_to_extracted_frames,
                js,
                sort_keys=True,
                indent=4,
            )
        logging.info(
            f"New json file with extracted frames saved at {json_output_file}"
        )

    # Save suggested frames as png files (extraction with opencv)
    extract_frames_to_label_from_video(
        map_videos_to_extracted_frames,
        output_subdir_path,
        flag_parent_dir_subdir_in_output=False,
    )


def app_wrapper():
    """Wrap function for the Typer app."""
    app()


if __name__ == "__main__":
    app_wrapper()
