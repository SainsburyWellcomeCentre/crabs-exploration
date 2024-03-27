# %%
# imports
import matplotlib.pyplot as plt
from torchvision.transforms import v2

from crabs.detection_tracking.datasets import CrabsCocoDetection
from crabs.detection_tracking.visualization import plot_sample

# %matplotlib qt #to pop out figures

# %%%%%%%%%%%%%%%%%%%
# Input data
IMG_DIR = "/Users/sofia/arc/project_Zoo_crabs/sep2023-full/frames"
ANNOT_FILE = "/Users/sofia/arc/project_Zoo_crabs/sep2023-full/annotations/VIA_JSON_combined_coco_gen.json"


# %%%%%%%%%%%%%%%%%%%%%%%%%%
# Create dataset
full_dataset = CrabsCocoDetection(
    [IMG_DIR],
    [ANNOT_FILE],
    transforms=None,
)

# %%%%%%%%%%%%%%%%%%%%%%%
# Sample a frame
sample = full_dataset[0]


# %%%%%%%%%%%%%%%
def transform_n_times_and_plot(sample, transform, n=1):
    transformed_imgs = [transform(sample) for _ in range(n)]

    plot_sample([sample] + transformed_imgs)
    plt.gcf().gca().set_title("transform")
    return transformed_imgs


# %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
# Random rotation

transform = v2.RandomRotation(degrees=(-10, 10))
transform_n_times_and_plot(sample, transform)


# %%
