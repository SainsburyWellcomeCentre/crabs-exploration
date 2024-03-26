import pytest
from unittest.mock import patch, MagicMock
from crabs.detection_tracking.datamodules import CrabsDataModule

@pytest.fixture
def crabs_data_module():
    return CrabsDataModule(
        list_img_dirs=["dir1", "dir2"],
        list_annotation_files=["anno1", "anno2"],
        config={"train_fraction": 0.8, "val_over_test_fraction": 0.2},
        split_seed=123
    )

# def test_get_train_transform(crabs_data_module):
#     train_transform = crabs_data_module._get_train_transform()
#     # Write assertions for the train transform

# def test_get_test_val_transform(crabs_data_module):
#     test_val_transform = crabs_data_module._get_test_val_transform()
#     # Write assertions for the test_val transform

# def test_collate_fn(crabs_data_module):
#     batch = ([1, 2, 3], [4, 5, 6])
#     collated_batch = crabs_data_module._collate_fn(batch)
#     # Write assertions for the collated batch

# def test_compute_splits(crabs_data_module):
#     train_dataset, test_dataset, val_dataset = crabs_data_module._compute_splits()
#     # Write assertions for the computed datasets

# @patch("crabs.CrabsCocoDetection")
# def test_prepare_data(mock_coco_detection):
#     crabs_data_module = CrabsDataModule(
#         list_img_dirs=["dir1", "dir2"],
#         list_annotation_files=["anno1", "anno2"],
#         config={"train_fraction": 0.8, "val_over_test_fraction": 0.2},
#         split_seed=123
#     )
#     crabs_data_module.prepare_data()
#     # Write assertions for prepare_data method

# @patch("crabs.DataLoader")
# def test_train_dataloader(mock_dataloader, crabs_data_module):
#     train_dataloader = crabs_data_module.train_dataloader()
#     # Write assertions for train_dataloader

# @patch("crabs.DataLoader")
# def test_val_dataloader(mock_dataloader, crabs_data_module):
#     val_dataloader = crabs_data_module.val_dataloader()
#     # Write assertions for val_dataloader

# @patch("crabs.DataLoader")
# def test_test_dataloader(mock_dataloader, crabs_data_module):
#     test_dataloader = crabs_data_module.test_dataloader()
#     # Write assertions for test_dataloader
