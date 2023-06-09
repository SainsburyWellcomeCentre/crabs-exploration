# %%
# OpenCV tutorial
#   https://docs.opencv.org/4.x/d9/df8/tutorial_root.html
# A nice tutorial
#   - https://note.nkmk.me/en/python-opencv-videocapture-file-camera/
# 
# FFMPEG-python looks useful for cropping
#   - https://github.com/kkroening/ffmpeg-python
# %%
import cv2
import pathlib as pl

# %%%%%%%%%%%%%%%%%%%%%%
# Data
######################
# video_path = '/Users/sofia/Documents_local/project_Zoo_crabs/crab_sample_data/NINJAV_S001_S001_T001.MOV'
# video_path = '/Users/sofia/Documents_local/project_Zoo_crabs/crab_sample_data/NINJAV_S001_S001_T003_subclip.mp4'
# 6min clip ---> 25 Hz

videos_parent_dir = pl.Path('/Volumes/zoo/raw/CrabField/ramalhete_Sep21')
file_types = ('**/*.MOV', '**/*.mp4', '**/*.avi')
list_paths = []
for typ in file_types:
    list_paths.extend(
        [p for p in list(videos_parent_dir.glob(typ)) if not p.name.startswith('._')]
    )


flag_save_sample_frames = False
# %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
# Video props and sample frames
###############################
map_file_to_props = {}

for p in list_paths:
    # filename
    fileID = str(pl.Path(*p.parts[-2:]))

    # create video capture object
    cap = cv2.VideoCapture(str(p))
    print(cap.isOpened())  # check it is correctly open (VideoCapture doesnt check)

    if cap.isOpened():
        print(f"{fileID}")
    else:
        print(f"{fileID} skipped....")
        continue

    # get video params
    nframes = cap.get(cv2.CAP_PROP_FRAME_COUNT)  # 9000
    width = cap.get(cv2.CAP_PROP_FRAME_WIDTH)  # 1920.0
    height = cap.get(cv2.CAP_PROP_FRAME_HEIGHT)  # 1080.0
    frame_rate = cap.get(cv2.CAP_PROP_FPS)  # 25fps

    map_file_to_props[fileID] = {
        'nframes': nframes,
        'size': (width, height),
        'frame_rate': frame_rate
    }

    if flag_save_sample_frames:
        # save frame at 50% of the total length
        # cap.set(cv2.CAP_PROP_POS_FRAMES, int(nframes / 2)) ---> this didnt seem to change it?
        total_ms = (
            cap.get(cv2.CAP_PROP_FRAME_COUNT) / cap.get(cv2.CAP_PROP_FPS)
        ) * 1000
        print(cap.set(cv2.CAP_PROP_POS_MSEC, 0.5 * total_ms)) 

        success, frame = cap.read()
        if success:
            file_path = 'sample_frames' / pl.Path(
                '_'.join((p.parts[-2], p.stem)) + '_' +
                f"frame{int(cap.get(cv2.CAP_PROP_POS_FRAMES))}.png"
            )
            cv2.imwrite(
                    str(file_path),
                    frame
            )  # save frame as JPEG file 

# %%%%%%%%%%%%%%%%%%%%%%%
# Print video properties
###########################
k0 = list(map_file_to_props.keys())[0]
inner_keys = map_file_to_props[k0].keys()

for ik in inner_keys:
    for k in sorted(map_file_to_props.keys()):
        print(f"{k}, {ik}: {map_file_to_props[k][ik]}")
        # print(f"{map_file_to_props[k][ik]},")
print('-----')

# %%%%%%%%%%%%%%%%%%%%%%%%%%%%%
# Save a clip of selected file
#####################

video_file = pl.Path(sorted(map_file_to_props.keys())[0])

# initialise capture and videowriter
cap = cv2.VideoCapture(str(videos_parent_dir / video_file))
fourcc = cv2.VideoWriter_fourcc('m', 'p', '4', 'v')
videowriter = cv2.VideoWriter(
    '_'.join((video_file.parts[-2], video_file.stem)) + '_clip.mp4', 
    fourcc, 
    map_file_to_props[str(video_file)]['frame_rate'], 
    tuple(int(x) for x in map_file_to_props[str(video_file)]['size'])
)

print(cap.get(cv2.CAP_PROP_POS_FRAMES)) # ensure we start at 0

# %%%%%%%%%%%%%%%%%%%
clip_length_in_s = 6*60
end_frame = cap.get(cv2.CAP_PROP_FPS)*clip_length_in_s - 1 # 0-indexed

print(cap.get(cv2.CAP_PROP_POS_FRAMES)) # ensure we start at 0
print(end_frame)

count = 0
while count <= end_frame:
    success_frame, frame = cap.read()
    if not success_frame:
        print("Can't receive frame. Exiting ...")
        break    
    videowriter.write(frame)
    count += 1

# Release everything if job is finished
cap.release()
videowriter.release()
cv2.destroyAllWindows()


# %%%%%%%%%%%
# # Current number of frames and elapsed time are 0 when opened
# print(cap.get(cv2.CAP_PROP_POS_FRAMES))
# print(cap.get(cv2.CAP_PROP_POS_MSEC))


# %%%%%%%%%%%%%%%%%%%
# Read method

# select midframe
# cap.set(cv2.CAP_PROP_POS_FRAMES)

# check
print(cap.get(cv2.CAP_PROP_POS_FRAMES))
print(cap.get(cv2.CAP_PROP_POS_MSEC))

# check channels per frame
ret, frame = cap.read() # returns a tuple: bool indicating if the frame was successfully read or not and ndarray, the array of the image.

# print(ret)
# print(type(frame))  # numpy array
# print(frame.shape)  # 3 channels


# see here
# https://stackoverflow.com/a/63519593
cv2.imshow("video", frame)
cv2.waitKey(0)
cv2.destroyAllWindows()
cv2.waitKey(1) # click any key to close while window is active
# %%
