from typing import Optional
from rich import print as rprint
from rich.console import Console


from pytorch_lightning.loggers import WandbLogger
from transformers.utils import logging

from ..config import Config


def get_loggers(config: Config):
    # set huggingface verbosity to ERROR
    logging.set_verbosity(verbosity=logging.ERROR)

    loggers = []
    if config.logger.wandb:
        wandb_logger = WandbLogger(
            offline=config.logger.wandb_offline, project=config.logger.wandb_project
        )
        wandb_logger.experiment.config.update(config)
        loggers.append(wandb_logger)

    return loggers


def spinner_animation(message: str, spinner_type: Optional[str] = "dots"):
    def decorator(func):
        def wrapper(*args, **kwargs):
            console = Console()
            with console.status(message, spinner=spinner_type):
                return func(*args, **kwargs)

        return wrapper

    return decorator
