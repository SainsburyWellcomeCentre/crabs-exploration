import sys

import torch

sys.path.append('RAFT/core')

import argparse
from pathlib import Path

import cv2
import numpy as np
import torch
from RAFT.core.raft import RAFT
from RAFT.core.utils import flow_viz
from RAFT.core.utils.utils import InputPadder

# -------------------
# Set device
# -------------------
if torch.cuda.is_available():
    DEVICE = 'cuda'
elif torch.backends.mps.is_available():
    DEVICE = 'mps'
else:
    DEVICE = 'cpu'


# -------------------
# aux class and fns
# -------------------
class dotdict(dict):
    """dot.notation access to dictionary attributes
    # https://stackoverflow.com/questions/2352181/how-to-use-a-dot-to-access-members-of-dictionary
    """
    __getattr__ = dict.get
    __setattr__ = dict.__setitem__
    __delattr__ = dict.__delitem__


def opencv_cap_to_torch_tensor(opencv_cap_frame):
    # opencv cap returns numpy array of size (h, w, channel)
    img = np.array(opencv_cap_frame).astype(np.uint8) # TODO: it is already np.array do I need to make it np.unit8?
    img = torch.from_numpy(img).permute(2, 0, 1).float()
    return img[None].to(DEVICE)


# -------------------
# core fn
# -------------------
def run_model_on_video(args):
    # -------------------
    # initialise model
    # -------------------
    model = torch.nn.DataParallel(RAFT(args))
    if DEVICE in ['cpu', 'mps']:
        model.load_state_dict(torch.load(args.model, map_location=DEVICE))
    else:
        model.load_state_dict(torch.load(args.model))

    model = model.module
    model.to(DEVICE)
    model.eval()

    # -----------------------------------------
    # initialise video capture
    # -----------------------------------------
    cap = cv2.VideoCapture(str(args.input_data))

    nframes = cap.get(cv2.CAP_PROP_FRAME_COUNT)  # 9000
    width = cap.get(cv2.CAP_PROP_FRAME_WIDTH)  # 1920.0
    height = cap.get(cv2.CAP_PROP_FRAME_HEIGHT)  # 1080.0
    frame_rate = cap.get(cv2.CAP_PROP_FPS)  # 25fps

    print(f'n frames: {nframes}')
    print(f'size: {(width, height)}')
    print(f'frame rate: {frame_rate}')

    # -----------------------------------------
    # initialise video writer
    # -----------------------------------------
    # create output dir if it doesnt exist
    videowriter_path = Path(args.output_dir) / Path(
        Path(args.input_data).stem + '_flow.mp4'
    )
    videowriter_path.mkdir(parents=True, exist_ok=True)

    # initialise videowriter
    videowriter = cv2.VideoWriter(
        str(videowriter_path), 
        cv2.VideoWriter_fourcc('m', 'p', '4', 'v'), 
        frame_rate, 
        tuple(int(x) for x in (width, height))
    )


    # -----------------------------------------
    # run inference in every frame
    # -----------------------------------------
    # mmmm quite slow.....
    with torch.no_grad():

        # ensure we start capture at 0
        if cap.get(cv2.CAP_PROP_POS_FRAMES) == 0: 
            frame_idx = 0
            frame_idx_stop = int(nframes/args.step_frames)*args.step_frames

            while frame_idx < frame_idx_stop:
                # set 'index to read next' to the desired position
                cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
                print(f'focus frame: {cap.get(cv2.CAP_PROP_POS_FRAMES)}')

                # read frame f
                success_frame_1, frame_1 = cap.read()
                # the index to read next is now at frame 2, 
                # set it as the next starting position
                frame_idx = cap.get(cv2.CAP_PROP_POS_FRAMES)

                # read frame at f+n
                cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx + args.step_frames)
                success_frame_2, frame_2 = cap.read()
                # OJO index to read next is now at f+n+1, but we will reset it 
                # to frame_idx at the start of the next iteration

                # if at least one of them is not successfully read, exit
                if not any([success_frame_1, success_frame_2]):
                    print("At least one frame was not read correctly. Exiting ...")
                    break    

                # make them torch tensors
                image1 = opencv_cap_to_torch_tensor(frame_1) 
                # In example: output is torch tensor of 
                # size([1, 3, 436, 1024]) (h x w)
                image2 = opencv_cap_to_torch_tensor(frame_2)

                # pad images (why?)---------
                padder = InputPadder(image1.shape)
                image1, image2 = padder.pad(image1, image2)

                # compute flow
                # output has batch channel on the first dim
                # TODO: what is flow_low? (downsampled?)
                flow_low, flow_up = model(
                    image1, 
                    image2, 
                    iters=20, 
                    test_mode=True
                )

                # convert output to numpy array and reorder channels
                # first dim = batch size (1)
                # second, third, fourth =  color channel, height, width
                flow_uv = flow_up[0].permute(1, 2, 0).cpu().numpy() 
                # permute: c x h x w ---> h x w x c
                
                # map optical flow to rgb image pixel space
                # ATT opencv uses BGR for color order! 
                flow_colorwheel_bgr = flow_viz.flow_to_image(
                    flow_uv, 
                    convert_to_bgr=True  # if False (default) this function would convert the output to RGB
                )
             
                # add to videowriter (as an opencv function, it expects BGR input)
                videowriter.write(flow_colorwheel_bgr)
        else:
            print('Starting frame index different from 0, closing without reading...')

        # -----------------------------------------
        # release capture and close video writer
        # -----------------------------------------
        cap.release()
        videowriter.release()


if __name__ == '__main__':

    # ------------------------------------
    # parse command line arguments
    # -------------------
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', help="restore checkpoint")
    parser.add_argument('--input_data', help="dataset for evaluation")
    parser.add_argument('--output_dir', help="directory to save output to")
    parser.add_argument('--small', action='store_true', help='use small model')
    parser.add_argument('--mixed_precision', action='store_true', help='use mixed precision')
    parser.add_argument('--alternate_corr', action='store_true', help='use efficent correlation implementation')
    parser.add_argument('--step_frames', type=int, nargs='?', const=1, default=1, help='compute optical flow between these number of frames')
    args = parser.parse_args()

    run_model_on_video(args)