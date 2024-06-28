"""Utils used in training"""

import logging
from typing import Optional

import torch

DEFAULT_ANNOTATIONS_FILENAME = "VIA_JSON_combined_coco_gen.json"


def get_checkpoint_type(checkpoint_path: Optional[str]) -> Optional[str]:
    """Get checkpoint type (full or weights) from the checkpoint path."""
    checkpoint = torch.load(checkpoint_path)  # fails if path doesn't exist
    if all(
        [
            param in checkpoint
            for param in ["optimizer_states", "lr_schedulers"]
        ]
    ):
        checkpoint_type = "full"  # for resuming training
        logging.info(
            f"Resuming training from checkpoint at: {checkpoint_path}"
        )
    else:
        checkpoint_type = "weights"  # for fine tuning
        logging.info(
            f"Fine-tuning training from checkpoint at: {checkpoint_path}"
        )

    return checkpoint_type


def log_data_augm_as_artifacts(logger, data_module):
    """Log data augmentation transforms as artifacts in MLflow."""
    for transform_str in ["train_transform", "test_val_transform"]:
        logger.experiment.log_text(
            text=str(getattr(data_module, f"_get_{transform_str}")()),
            artifact_file=f"{transform_str}.txt",
            run_id=logger.run_id,
        )
