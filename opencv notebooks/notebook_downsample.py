# %%
# https://www.tutorialkart.com/opencv/python/
# To check how much we can downsample features to select frames to label...
import pathlib as pl

import cv2

from matplotlib import pyplot as plt

# %%%%%%%%%%%%%%%%%%%%%%%%%

sample_frame = pl.Path(
    "/Users/sofia/Documents_local/project_Zoo_crabs/"
    "crabs-exploration/pose_estimation_4k/output/"
    "Camera2_NINJAV_S001_S001_T010_frame_008455.png"
)

# %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

# Read original image
img_og = cv2.imread(str(sample_frame))
size_og = (img_og.shape[1], img_og.shape[0])  # we want width, height!

# cv2.imshow("image", img_og)
# cv2.waitKey(0)
# cv2.destroyAllWindows()
# cv2.waitKey(1) # click any key to close while window is active

# %%

window_id = "img"  # f'scale {f}'
cv2.namedWindow(window_id, cv2.WINDOW_NORMAL)
cv2.resizeWindow(window_id, size_og[0], size_og[1])

# Read downsampled
list_downsampled_factors = [1, 0.75, 0.5, 0.25]
list_downsampled_imgs = []
for f in list_downsampled_factors:
    size_down = tuple(int(s * f) for s in size_og)
    img_downsampled = cv2.resize(img_og, size_down, interpolation=cv2.INTER_NEAREST)

    window_id = f"scale {f}"
    cv2.namedWindow(window_id, cv2.WINDOW_NORMAL)

    cv2.imshow(window_id, img_downsampled)
    cv2.resizeWindow(window_id, size_og[0], size_og[1])
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    cv2.waitKey(1)

    # append to list
    list_downsampled_imgs.append(img_downsampled)


# %%%%%%%%%%%%%%%%%%%%%%%%%%%%%
list_downsampled_imgs.append(img_og)
for im in list_downsampled_imgs:
    img_downsampled_color = cv2.cvtColor(im, cv2.COLOR_BGR2RGB)
    plt.figure()
    plt.imshow(img_downsampled_color)
    plt.title(f"scale {f}")
    plt.show()


# %%
# %%
