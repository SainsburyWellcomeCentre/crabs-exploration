# %%
import yaml  # type: ignore

from crabs.detector.datamodules import CrabsDataModule
from crabs.detector.utils.visualization import plot_sample

# %%%%%%%%%%%%%%%%%%%
# Input data
IMG_DIR = "/home/sminano/swc/project_crabs/data/sep2023-full/frames"
ANNOT_FILE = "/home/sminano/swc/project_crabs/data/sep2023-full/annotations/VIA_JSON_combined_coco_gen.json"
CONFIG = "/home/sminano/swc/project_crabs/crabs-exploration/crabs/detection_tracking/config/faster_rcnn.yaml"
SPLIT_SEED = 42

# %%%%%%%%%%%%%%%%%%%%
# Read config as dict
with open(CONFIG, "r") as f:
    config_dict = yaml.safe_load(f)

# %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
# Create datamodule for the input data
dm = CrabsDataModule(
    list_img_dirs=[IMG_DIR],
    list_annotation_files=[ANNOT_FILE],
    config=config_dict,
    split_seed=SPLIT_SEED,
)
# %%%%%%%%%%%%%%%%%%%%%%%%
# Setup for train / test
dm.prepare_data()
dm.setup("fit")


# %%%%%%%%%%%%%%%%%%%%%%%%%%%
# after this: dm.train_dataset should have transforms, (but not dm.test_dataset)
print(dm.train_transform)
print(dm.val_transform)
print(dm.test_transform)

# %%%%%%%%%%%%%%%%%%%%%%%%%
# visualize
train_dataset = dm.train_dataset
train_sample = train_dataset[0]
plot_sample([train_sample])

# %%
