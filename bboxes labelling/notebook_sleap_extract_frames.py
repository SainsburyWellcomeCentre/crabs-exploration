# 1- Get list of suggested frames from sleap
# 2- Extract those frames with opencv and save to ceph
# https://github.com/talmolab/sleap/blob/81b43425e98ab43a155eb2f3a46910d51e73ca61/sleap/info/feature_suggestions.py#L550

# %%
from sleap import Video
from sleap.info.feature_suggestions import FeatureSuggestionPipeline, ParallelFeaturePipeline

import cv2
from pathlib import Path

# %%%%%%%%%%%%%%%%%%%
# Input data --- arg parser?
vids = [
    Video.from_filename("/Users/sofia/Documents_local/project_Zoo_crabs/crabs-exploration/crab_sample_data/sample_clips/cam1_NINJAV_S001_S001_T001_clip.mp4"),
    Video.from_filename("/Users/sofia/Documents_local/project_Zoo_crabs/crabs-exploration/crab_sample_data/sample_clips/NINJAV_S001_S001_T003_subclip.mp4"),
]

output_path = Path('/tmp/')

# %%%%%%%%%%%%%%%%%%%%%%%%%%%
# Define frame extraction pipeline
pipeline = FeatureSuggestionPipeline(
    per_video=200,  # frames per video to use as input
    sample_method="stride",
    scale=1.0,
    feature_type="raw",
    brisk_threshold=80,  #?
    n_components=5,
    n_clusters=5,
    per_cluster=5,
)

suggestions = ParallelFeaturePipeline.run(
    pipeline, 
    vids, 
    parallel=True
)
#---I think I need parallel=True to compute features per video, is that correct?
# yes: https://github.com/talmolab/sleap/blob/1a0404c0ffae7b248eb360562b0bb95a42a287b6/sleap/gui/suggestions.py#L159
#  

# filter suggestions: only applicable if already labelled frames in the set, right?
# suggestions = VideoFrameSuggestions.filter_unique_suggestions(
#         labels, vids, proposed_suggestions
#     )

print(suggestions)  
# list of suggestions per frame (is it deterministic for the params above?) -- check with GUI
# each element of list is a SuggestionFrame object with
# - .video: video frame belongs to: suggestions[0].video.backend.filename
# - .frame_idx: frame index suggestions[0].frame_idx
# - .group: cluster?

# TODO: how to request features per video?
# TODO: are frames 0-indexed?
# %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
# Extract suggested frames with opencv
map_videos_to_extracted_frames = {}
for vid in vids:
    vid_str = vid.backend.filename
    map_videos_to_extracted_frames[vid_str] = [
        sugg.frame_idx
        for sugg in suggestions
        if sugg.video.backend.filename == vid_str
    ]


for vid_str in map_videos_to_extracted_frames.keys():
    # initialise opencv capture
    cap = cv2.VideoCapture(vid_str)
    
    # check
    if cap.isOpened():
        print(f"{Path(vid_str).stem}")
    else:
        print(f"{Path(vid_str).stem} skipped....")
        continue

    # go to specified frames
    for frame_idx in map_videos_to_extracted_frames[vid_str]:

        # read frame
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
        success, frame = cap.read()

        # save to file
        if not success:
            print('error reading frame..')
            break
        else:
            file_path = output_path / Path(vid_str).with_suffix('') / Path(
                f'frame_{frame_idx}.png'
            )
            cv2.imwrite(
                file_path,
                frame
            )


    # close capture
# %%
