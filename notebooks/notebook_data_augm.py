# %%
import yaml  # type: ignore

from crabs.detection_tracking.datamodules import CrabsDataModule
from crabs.detection_tracking.visualization import plot_sample

# %%%%%%%%%%%%%%%%%%%
# Input data
IMG_DIR = "/Users/sofia/arc/project_Zoo_crabs/sep2023-full/frames"
ANNOT_FILE = "/Users/sofia/arc/project_Zoo_crabs/sep2023-full/annotations/VIA_JSON_combined_coco_gen.json"
CONFIG = "/Users/sofia/arc/project_Zoo_crabs/restructure/crabs-exploration/crabs/detection_tracking/config/faster_rcnn.yaml"
SPLIT_SEED = 42

# %%%%%%%%%%%%%%%%%%%%
# Read config as dict
with open(CONFIG, "r") as f:
    config_dict = yaml.safe_load(f)

# %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
# Create datamodule for the input data
dm = CrabsDataModule(
    [IMG_DIR],
    [ANNOT_FILE],
    config_dict,
    SPLIT_SEED,
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

# %%
train_dataset = dm.train_dataset
train_sample = train_dataset[0]
plot_sample([train_sample])

# %%
