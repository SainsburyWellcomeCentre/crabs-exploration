'''
A script to extract frames for labelling using SLEAP's algorith,

Example usage:
    python bboxes\ labelling/extract_frames_to_label_w_sleap.py 
    'crab_sample_data/sample_clips/' 
    --initial_samples 5 
    --n_components 2 
    --n_clusters 2 
    --per_cluster 1 
    --compute_features_per_video

TODO: can I make it deterministic?
TODO: check https://github.com/talmolab/sleap-io/tree/main/sleap_io
TODO: change it to copy directory structure from input?
'''

import argparse
import json
from datetime import datetime
from pathlib import Path

import cv2
from sleap import Video
from sleap.info.feature_suggestions import (FeatureSuggestionPipeline,
                                            ParallelFeaturePipeline)

import logging

# ------------------
# Utils
# -----------------
def get_sleap_videos_list(
    list_video_locations: list[str],  
    list_video_extensions: list = ['mp4']
):
    # split locations between files and directories
    list_video_paths = []
    for loc in list_video_locations:
        location_path = Path(loc)

        # if dir: look for files with any of the relevant extensions 
        # (only one level in)
        if location_path.is_dir():
            for ext in list_video_extensions:
                list_video_paths.extend(location_path.glob(f'[!.]*.{ext}'))  
                # exclude hidden files
        # if file has the relevant extension: append directly to list?
        elif location_path.is_file() and (
            location_path.suffix[1:] in list_video_extensions
            # suffix includes dot
        ):
            list_video_paths.append(location_path)

    # transform list of videos to sleap videos,
    # filtering out those that opencv cannot open
    # TODO is there a better way to do this?
    list_sleap_videos = []
    for vid_path in list_video_paths:
        cap = cv2.VideoCapture(str(vid_path))
        if cap.isOpened():
            list_sleap_videos.append(
                Video.from_filename(str(vid_path))
            )
            cap.release()

    # print warning if list is empty
    if not list_sleap_videos:
        logging.warning(
            "List of videos is empty \n" 
            f"\t locations:{list_video_locations}\n "
            f"\t extensions:{list_video_extensions})\n"
        )

    return list_sleap_videos


def get_map_videos_to_extracted_frames(
    list_sleap_videos,
    suggestions
):
    map_videos_to_extracted_frames = {}
    for vid in list_sleap_videos:
        vid_str = vid.backend.filename
        map_videos_to_extracted_frames[vid_str] = sorted(
            [
                sugg.frame_idx
                for sugg in suggestions
                if sugg.video.backend.filename == vid_str
            ]
        )
    return map_videos_to_extracted_frames


# ------------------
# main
# -----------------
def extract_frames_to_label(args):

    # -------------------------------------------------------
    # Run frame extraction pipeline from SLEAP
    # -------------------------------------------------------
    # read videos as sleap Video instances
    list_sleap_videos = get_sleap_videos_list(
        args.list_video_locations,
        args.video_extensions
    )

    # define the pipeline
    pipeline = FeatureSuggestionPipeline(
        per_video=args.initial_samples,  
        sample_method=args.sample_method, 
        scale=args.scale, 
        feature_type=args.feature_type,  
        n_components=args.n_components,  
        n_clusters=args.n_clusters, 
        per_cluster=args.per_cluster,
    )

    # run the pipeline (per video, if args.compute_features_per_video=True)
    suggestions = ParallelFeaturePipeline.run(
        pipeline, 
        list_sleap_videos, 
        parallel=args.compute_features_per_video,
    )

    # sleap frames are 0-indexed (right?)
    map_videos_to_extracted_frames = get_map_videos_to_extracted_frames(
        list_sleap_videos,
        suggestions
    )

    # --------------------
    # Prepare output data
    # ----------------------
    # create timestamp folder inside output folder if it doesnt exist
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    output_dir_timestamped = Path(args.output_path) / f'{timestamp}' 
    output_dir_timestamped.mkdir(parents=True, exist_ok=True)

    # save extracted frames as json file
    json_output_file = output_dir_timestamped / 'extracted_frames.json'
    with open(json_output_file, 'w') as js:
        json.dump(
            map_videos_to_extracted_frames, 
            js,
            sort_keys=True, 
            indent=4,
    )

    # -------------------------------------------------------
    # Extract suggested frames with opencv
    # -------------------------------------------------------

    # loop thru videos and extract frames
    for vid_str in map_videos_to_extracted_frames.keys():
        # initialise opencv capture
        cap = cv2.VideoCapture(vid_str)
        
        # check
        logging.info('---------------------------')
        if cap.isOpened():
            logging.info(f"Processing {Path(vid_str)}")
        else:
            logging.info(f"Error processing {Path(vid_str)}, skipped....")
            continue

        # create video output dir inside timestamped one
        video_output_dir = (
            output_dir_timestamped  #/   # timestamp
            # Path(vid_str).parent.stem /  # parent dir of input video
            # Path(vid_str).stem  # video name
        )
        video_output_dir.mkdir(parents=True, exist_ok=True)

        # go to the selected frames
        for frame_idx in map_videos_to_extracted_frames[vid_str]:

            # read frame
            # OJO in opencv, frames are 0-index, and I *think* in sleap too?
            cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
            success, frame = cap.read()

            # save to file
            if not success or frame is None:
                raise KeyError(f"Unable to load frame {frame_idx} from {vid_str}.")

            else:
                file_path = video_output_dir / Path(
                    f"{Path(vid_str).parent.stem}_"
                    f"{Path(vid_str).stem}_"
                    f"frame_{frame_idx:06d}.png"
                )
                img_saved = cv2.imwrite(
                    str(file_path),
                    frame
                )
                if img_saved:
                    logging.info(f"frame {frame_idx} saved at {file_path}")
                else:
                    logging.info(f"ERROR saving {Path(vid_str).stem}, frame {frame_idx}...skipping")
                    continue


        # close capture
        cap.release()


if __name__ == '__main__':    
    # ------------------------------------
    # parse command line arguments
    # ------------------------------

    # TODO: add grayscale option?
    # TODO: read extracted frames from file?
    parser = argparse.ArgumentParser()
    parser.add_argument('list_video_locations',   # positional, needs first fwd slash!
                        nargs='*',
                        help="list of paths to directories with videos, or to specific video files")
    parser.add_argument('--output_path',
                        default='.',  # does this work?
                        help=(
                            "path to directory in which to store extracted" 
                            " frames (by default, the current directory)"
                        )
                        )
    parser.add_argument('--video_extensions', 
                        nargs='*',
                        default=['mp4'],
                        help="extensions to search for when looking for video files")
    parser.add_argument('--initial_samples', 
                        type=int, 
                        nargs='?', 
                        default=200, 
                        help='initial number of frames to extract per video')
    parser.add_argument('--sample_method',
                        type=str,
                        default='stride',
                        choices=['random', 'stride'],
                        help='method to sample initial frames')  # ok?
    parser.add_argument('--scale', 
                        type=float, 
                        nargs='?', 
                        default=1.0, 
                        help='factor to apply to the images prior to PCA and k-means clustering')  # help ok?
    parser.add_argument('--feature_type', 
                        type=str,
                        nargs='?',
                        default='raw',
                        choices=['raw', 'brisk', 'hog'],
                        help='type of input feature') 
    parser.add_argument('--n_components', 
                        type=int,
                        nargs='?',
                        default=5,
                        help='number of PCA components')
    parser.add_argument('--n_clusters', 
                        type=int, 
                        nargs='?', 
                        default=5, 
                        help='number of k-means clusters')
    parser.add_argument('--per_cluster', 
                        type=int, 
                        nargs='?', 
                        default=5, 
                        help='number of frames to sample per cluster')
    parser.add_argument('--compute_features_per_video', 
                        type=bool, 
                        nargs='?', 
                        const=True,
                        help='whether to compute the (PCA?) features per video, or across all videos')
    # parser.add_argument('--random_seed', 
    #                     type=int, 
    #                     nargs='?',
    #                     default=42, 
    #                     help='random seed')
    args = parser.parse_args()

    # ------------------------
    # run frame extraction
    # ------------------------
    extract_frames_to_label(args)