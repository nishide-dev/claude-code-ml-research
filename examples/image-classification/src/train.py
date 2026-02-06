"""Training script for CIFAR-10 image classification."""

import logging
from pathlib import Path

import hydra
import pytorch_lightning as pl
from hydra.utils import instantiate
from omegaconf import DictConfig, OmegaConf
from pytorch_lightning import seed_everything

logger = logging.getLogger(__name__)


@hydra.main(version_base=None, config_path="../configs", config_name="config")
def main(cfg: DictConfig) -> None:
    """Main training function.

    Args:
        cfg: Hydra configuration
    """
    # Print config
    logger.info("Configuration:\n%s", OmegaConf.to_yaml(cfg))

    # Set seed for reproducibility
    if "seed" in cfg:
        seed_everything(cfg.seed, workers=True)
        logger.info("Seed set to: %d", cfg.seed)

    # Create output directory
    output_dir = Path(cfg.get("paths", {}).get("output_dir", "./outputs"))
    output_dir.mkdir(parents=True, exist_ok=True)

    # Instantiate data module
    logger.info("Instantiating data module: %s", cfg.data._target_)
    datamodule: pl.LightningDataModule = instantiate(cfg.data)

    # Instantiate model
    logger.info("Instantiating model: %s", cfg.model._target_)
    model: pl.LightningModule = instantiate(cfg.model)

    # Instantiate logger(s)
    lightning_logger = None
    if "logger" in cfg:
        logger.info("Instantiating logger(s)")
        lightning_logger = []
        for logger_name, logger_cfg in cfg.logger.items():
            if logger_cfg is not None and "_target_" in logger_cfg:
                lightning_logger.append(instantiate(logger_cfg))
                logger.info("  - %s", logger_name)

    # Instantiate trainer
    logger.info("Instantiating trainer: %s", cfg.trainer._target_)
    trainer: pl.Trainer = instantiate(cfg.trainer, logger=lightning_logger)

    # Log hyperparameters
    if lightning_logger:
        for lg in lightning_logger if isinstance(lightning_logger, list) else [lightning_logger]:
            lg.log_hyperparams(OmegaConf.to_container(cfg, resolve=True))

    # Train the model
    logger.info("Starting training!")
    trainer.fit(model, datamodule=datamodule)

    # Test the model
    logger.info("Starting testing!")
    trainer.test(model, datamodule=datamodule, ckpt_path="best")

    # Print best checkpoint path
    if hasattr(trainer.checkpoint_callback, "best_model_path"):
        logger.info("Best checkpoint: %s", trainer.checkpoint_callback.best_model_path)

    logger.info("Training complete!")


if __name__ == "__main__":
    main()
