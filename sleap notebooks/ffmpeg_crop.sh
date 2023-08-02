#!/bin/bash

# lossless re enconding: https://stackoverflow.com/questions/33378548/ffmpeg-crop-a-video-without-losing-the-quality
INPUT_VIDEO="/Users/sofia/Documents_local/project_Zoo_crabs/pose estimation"
INPUT_VIDEO=$INPUT_VIDEO"/rendered videos/crabs_pose_4k_TD4/Camera2-NINJAV_S001_S001_T010.MOV.predictions.slp.avi"
INPUT_VIDEO_FILENAME="${INPUT_VIDEO%.*}"
INPUT_VIDEO_EXTENSION="${INPUT_VIDEO##*.}"

OUT_W=1196
OUT_H=410
X=1300
Y=1350

ffmpeg -i "$INPUT_VIDEO" \
    -filter:v "crop=$OUT_W:$OUT_H:$X:$Y" \
    -c:v libx264 -crf 0 -c:a copy \
    "$INPUT_VIDEO_FILENAME-crop.$INPUT_VIDEO_EXTENSION"
