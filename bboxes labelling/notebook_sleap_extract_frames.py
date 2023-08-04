# 1- Get list of suggested frames from sleap
# 2- Extract those frames with opencv and save to ceph
# https://github.com/talmolab/sleap/blob/81b43425e98ab43a155eb2f3a46910d51e73ca61/sleap/info/feature_suggestions.py#L550

# %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
import json
from datetime import datetime
from pathlib import Path

import cv2
from sleap import Video
from sleap.info.feature_suggestions import (
    FeatureSuggestionPipeline,
    ParallelFeaturePipeline,
)

# %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
# Input data --- arg parser?
video_dir = (
    "/Users/sofia/Documents_local/project_Zoo_crabs/crabs-exploration/"
    "crab_sample_data/sample_clips"
)
output_path = Path("/Users/sofia/Desktop/tmp/")
list_video_extensions = ["mp4"]
random_seed = 42

# also pass pipeline params as CLI inputs?
# initial_samples_per_video
# sample_method
# add grayscale option?

# %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
# Read videos as sleap Video instances
list_video_paths: list[Path] = []
for ext in list_video_extensions:
    list_video_paths.extend(Path(video_dir).glob(f"*.{ext}"))

list_sleap_videos = [
    Video.from_filename(str(vid_path)) for vid_path in list_video_paths
]


# %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
# Define frame extraction pipeline

# if random_seed:
#   random.seed(random_seed) ---> doesnt work

pipeline = FeatureSuggestionPipeline(
    per_video=20,  # frames per video to use as input
    sample_method="stride",
    scale=1.0,
    feature_type="raw",
    brisk_threshold=80,  # ?
    n_components=3,
    n_clusters=3,
    per_cluster=1,
)

suggestions = ParallelFeaturePipeline.run(
    pipeline,
    list_sleap_videos,
    parallel=False,
    # if parallel=True,
    # I get an error in notebook:
    # https://stackoverflow.com/questions/65859890/python-multiprocessing-with-m1-mac
)


print(suggestions)
# list of suggestions per frame (is it deterministic for the params above?)
#  -- check with GUI
# each element of list is a SuggestionFrame object with
# - .video: video frame belongs to: suggestions[0].video.backend.filename
# - .frame_idx: frame index suggestions[0].frame_idx
# - .group: cluster?

# TODO: how to request features per video?
# https://github.com/talmolab/sleap/blob/1a0404c0ffae7b248eb360562b0bb95a42a287b6/sleap/gui/suggestions.py#L159
# TODO: are frames 0-indexed?
# (from sleap code it looks like yes, but in the GUI they are 1-indexed)
# TODO: how to make frame extraction deterministic?
# %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
# Prepare output dir

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
output_path.mkdir(parents=True, exist_ok=True)
timestamp = datetime.now().strftime("%Y%m%d_%-H%M%-S")
json_output_file = output_path / f"frame_extraction_{timestamp}.json"
with open(json_output_file, "w") as js:
    json.dump(
        map_videos_to_extracted_frames,
        js,
        sort_keys=True,
        indent=4,
    )
# %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
# Extract suggested frames with opencv
# OJO in opencv frames are 0-indexed!
for vid_str in map_videos_to_extracted_frames:
    # initialise opencv capture
    cap = cv2.VideoCapture(vid_str)

    # check
    if cap.isOpened():
        print(f"{Path(vid_str).stem}")
    else:
        print(f"{Path(vid_str).stem} skipped....")
        continue

    # create output dir
    Path(output_path / Path(vid_str).stem).mkdir(parents=True, exist_ok=True)

    # go to specified frames
    for frame_idx in map_videos_to_extracted_frames[vid_str]:
        # read frame
        # OJO in opencv, frames are 0-index, and I *think* in sleap too?
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx - 1)
        success, frame = cap.read()

        # save to file
        if not success or frame is None:
            msg = f"Unable to load frame {frame_idx} from {vid_str}."
            raise KeyError(msg)

        else:
            file_path = (
                output_path / Path(vid_str).stem / Path(f"frame_{frame_idx:06d}.png")
            )
            img_saved = cv2.imwrite(str(file_path), frame)
            if img_saved:
                print(f"{Path(vid_str).stem}, frame {frame_idx} saved")
            else:
                print(
                    f"ERROR saving {Path(vid_str).stem}, "
                    f"frame {frame_idx}...skipping",
                )
                continue

    # close capture
    cap.release()


# %%
