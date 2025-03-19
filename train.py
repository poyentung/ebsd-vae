import logging
import os
from pathlib import Path
import random
from typing import Any, Mapping

import hydra
import torch
import numpy as np
import pytorch_lightning as pl
from hydra.utils import instantiate
from omegaconf import DictConfig

from src.lightning_module import VAELightningModule

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def maybe_instantiate(
    instance_or_config: Any | Mapping, expected_type=None, **kwargs
) -> Any:
    """Instantiates objects from configurations if needed.

    Args:
        instance_or_config: Either an instantiated object or a config dict with _target_.
        expected_type: Optional type to validate the instantiated object against.
        **kwargs: Additional arguments to pass to the instantiation function.

    Returns:
        The instantiated object or the original object if already instantiated.

    Raises:
        AssertionError: If expected_type is provided and the instance is not of that type.
    """
    if isinstance(instance_or_config, Mapping) and "_target_" in instance_or_config:
        instance = instantiate(instance_or_config, **kwargs)
    else:
        instance = instance_or_config
    assert expected_type is None or isinstance(instance, expected_type), (
        f"Expected {expected_type}, got {type(instance)}"
    )
    return instance


def set_random_seeds(seed: int) -> None:
    """Sets random seeds for reproducibility.

    Args:
        seed: The random seed to use.
    """
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)


def train(config: DictConfig) -> tuple[pl.Trainer, VAELightningModule]:
    """Trains a model using PyTorch Lightning.

    Args:
        config: A DictConfig containing configuration for the training run.

    Returns:
        A tuple containing the trainer and the trained lightning module.
    """
    # Set random seeds for reproducibility
    if config.seed is not None:
        set_random_seeds(config.seed)

    # Create directories
    save_dir = Path(config.trainer.logger.save_dir)
    try:
        os.makedirs(save_dir, exist_ok=True)
        check_point_path = save_dir / "checkpoints"
        check_point_path.mkdir(parents=True, exist_ok=True)
    except (PermissionError, FileNotFoundError) as e:
        logger.error(f"Failed to create directories: {e}")
        raise

    # Instantiate components
    logger.info(f"Instantiating trainer <{config.trainer._target_}>")
    trainer: pl.Trainer = maybe_instantiate(config.trainer, pl.Trainer)

    logger.info(f"Instantiating datamodule <{config.data_module._target_}>")
    datamodule: pl.LightningDataModule = maybe_instantiate(
        config.data_module, pl.LightningDataModule
    )

    logger.info(f"Instantiating lightning module <{config.lightning_module._target_}>")
    pl_module: VAELightningModule = maybe_instantiate(
        config.lightning_module, VAELightningModule
    )

    # Train the model
    trainer.fit(pl_module, datamodule=datamodule)

    return trainer, pl_module


@hydra.main(version_base="1.3", config_path="conf", config_name="train.yaml")
def main(config: DictConfig) -> None:
    """Entry point for the training script.

    Args:
        config: Configuration loaded by Hydra.
    """
    try:
        train(config)
    except Exception as e:
        logger.exception(f"Training failed: {e}")
        raise


if __name__ == "__main__":
    main()
