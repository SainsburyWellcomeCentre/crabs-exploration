# %%
from pathlib import Path

import napari
import numpy as np
import pandas as pd
from PIL import Image

# %%
images_dir = (
    "/Users/sofia/arc/project_Zoo_crabs/crab_loops_end_frames_slurm2764495"
)
prompts_dir = '/Users/sofia/arc/project_Zoo_crabs/burrow_prompts_per_day_20260423_143244'
# "/Users/sofia/arc/project_Zoo_crabs/burrow_prompts_per_video_20260423_153811"


# %%%%%%%%%%
class ImageArrayLazy:
    """A lazy array for images in a list."""

    def __init__(self, img_paths):
        self.img_paths = sorted(img_paths)
        # add image shape, assuming all have same as
        # first sample
        sample = np.array(Image.open(img_paths[0]))  # H, W, C
        self.img_h, self.img_w, self.img_c = sample.shape

    def __len__(self):
        return len(self.img_paths)

    def __getitem__(self, idx):
        return np.array(Image.open(self.img_paths[idx]))

    @property
    def shape(self):
        return (len(self.img_paths), self.img_h, self.img_w, self.img_c)
        # B, H, W, C


# %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
list_image_files = sorted(list(Path(images_dir).glob("*.png")))
image_array = ImageArrayLazy(list_image_files)


list_video_per_img = [
    img.stem.split("_", 1)[0].split("-Loop")[0]
    for img in image_array.img_paths
]

list_date_per_img = [video.split("-")[0] for video in list_video_per_img]

# %%%%%%%%%%%%%%%%%%%%%%%%%%
list_prompt_csv = sorted(list(Path(prompts_dir).glob("*.csv")))
list_df = []
for file in list_prompt_csv:
    list_df.append(pd.read_csv(file))

df_prompts = pd.concat(list_df)
df_prompts_points = df_prompts.drop(
    columns=[
        "prompt_bbox_xmin",
        "prompt_bbox_ymin",
        "prompt_bbox_xmax",
        "prompt_bbox_ymax",
    ]
)
df_prompts_bboxes = df_prompts.drop(
    columns=["prompt_point_x", "prompt_point_y"]
)

# %%
# df_prompts_points["napari_image_idx"] = df_prompts_points["group_id"].map(
#     list_video_per_img.index
# )

prompts_yx_per_group_id = {
    key: group[["prompt_point_y", "prompt_point_x"]].to_numpy()
    for key, group in df_prompts_points.groupby("group_id")
}


# map prompts to frames
list_zyx = []
start_idx = 0
for video_str, yx in prompts_yx_per_group_id.items():
    n_frames = sum([video_str == im for im in list_date_per_img]) # list_date_per_img? list_video_per_img
    zyx = np.concat(
        [
            np.c_[np.ones((yx.shape[0], 1)) * id + start_idx, yx]
            for id in range(n_frames)
        ],
        axis=0,
    )
    list_zyx.append(zyx)
    start_idx += n_frames

# %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
# Load frames in napari viewer

viewer = napari.Viewer()
viewer.add_image(np.asarray(image_array), name="image")


# %%%%%%%%%%%%%%%%%%%%%%%%
# Load prompts for SAM3

# frame, y, x
viewer.add_points(np.concat(list_zyx, axis=0))

# %%%%%%%%%%%%%%%%%%%%%
# Load SAM3 masks
# viewer.add_labels(np.asarray(mask_array), name=f"{LABEL_NAME} masks")

# %%
