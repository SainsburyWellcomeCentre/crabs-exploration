# %%
import cv2
import pathlib as pl


# %%%%%%%%%%%%%%%%%%%%%%%%%%%%%
# Save a clip of selected file

video_file = pl.Path('/Volumes/zoo/raw/CrabField/ramalhete_Sep21/cam2/NINJAV_S001_S001_T004.MOV')

clip_suffix = 'out_domain_sample'

frame_rate = 25
size = (1920, 1080)
# frames are considered 0-indexed here! (in sleap viewer they are 1-indexed)
# I split this video cam2_NINJAV_S001_S001_T003_subclip in two parts: 
# part 1: 0 - 7201
# part 2: 7211 - 9000
start_frame = 0  # part 1: 0; part 2: 7211
end_frame = 22500  # part 1: 7201; part 2: 9000

# %%%%%%%%%%%%%%%%%%%%%%%%%%%%%
# initialise capture and videowriter
cap = cv2.VideoCapture(str(video_file))
fourcc = cv2.VideoWriter_fourcc('m', 'p', '4', 'v')
videowriter = cv2.VideoWriter(
    '_'.join((video_file.parts[-2], video_file.stem)) + f'_{clip_suffix}.mp4', 
    fourcc, 
    frame_rate, 
    size
)

print(cap.get(cv2.CAP_PROP_POS_FRAMES)) # ensure we start at 0

# %%%%%%%%%%%%%%%%%%%
# clip_length_in_s = 6*60

print(cap.get(cv2.CAP_PROP_POS_FRAMES)) # ensure we start at 0
print(end_frame)

count = 0
while count <= end_frame:
    # read frame
    success_frame, frame = cap.read()
    

    # if not successfully read, exit
    if not success_frame:
        print("Can't receive frame. Exiting ...")
        break    

    # if frame within clip bounds: write to video
    if (count >= start_frame) and (count <= end_frame):
        videowriter.write(frame)

    count += 1

# Release everything if job is finished
cap.release()
videowriter.release()
cv2.destroyAllWindows()
# %%
