import numpy as np
import pytest
import torch
from PIL import Image
from torchvision import transforms

from crabs.detection_tracking.datamodule import (
    collate_fn,
    get_eval_transform,
    get_train_transform,
    myDataModule,
)


def test_collate_fn():
    batch = []
    collated = collate_fn(batch)
    assert collated is None

    # Test case 3: Test collating a batch with None values
    batch = [(1, 2), None, (6, 4)]
    collated = collate_fn(batch)
    assert collated == ((1, 6), (2, 4))


@pytest.fixture
def data_module(tmpdir):
    main_dir = tmpdir.mkdir("data")
    annotation = main_dir.join("annotation.json")
    annotation.write(
        '{"images": [{"id": 1, "file_name": "videofile_image1.jpg"}, '
        '{"id": 2, "file_name": "videofile_image2.jpg"}]}'
    )
    data_module_instance = myDataModule(
        str(main_dir), str(annotation), batch_size=2
    )
    data_module_instance.setup()
    return data_module_instance


def test_setup(data_module):
    data_module.setup()
    assert len(data_module.train_ids) > 0
    assert len(data_module.test_ids) > 0


def test_train_dataloader(data_module):
    # Test that train_dataloader returns a DataLoader object
    train_loader = data_module.train_dataloader()
    assert isinstance(train_loader, torch.utils.data.DataLoader)


def test_val_dataloader(data_module):
    # Test that val_dataloader returns a DataLoader object
    val_loader = data_module.val_dataloader()
    assert isinstance(val_loader, torch.utils.data.DataLoader)


def test_get_train_transform():
    transform = get_train_transform()

    assert isinstance(transform, transforms.Compose)

    mock_image = np.random.rand(3, 256, 256)
    mock_image = Image.fromarray(
        (mock_image * 255).astype(np.uint8).transpose(1, 2, 0)
    )

    transformed_image = transform(mock_image)

    assert transformed_image.shape == (3, 256, 256)
    assert torch.all(transformed_image >= 0) and torch.all(
        transformed_image <= 1
    )


def test_get_eval_transform():
    transform = get_eval_transform()

    assert isinstance(transform, transforms.Compose)

    mock_image = np.random.rand(3, 256, 256)
    mock_image = Image.fromarray(
        (mock_image * 255).astype(np.uint8).transpose(1, 2, 0)
    )

    transformed_image = transform(mock_image)

    assert transformed_image.shape == (3, 256, 256)
    assert torch.all(transformed_image >= 0) and torch.all(
        transformed_image <= 1
    )
