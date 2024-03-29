from typing import Any, Dict, Optional
from omegaconf import OmegaConf
from rich.console import Console

from pytorch_lightning.loggers import WandbLogger
from pytorch_lightning.callbacks import RichProgressBar
from pytorch_lightning.callbacks.progress.rich_progress import RichProgressBarTheme

from transformers.utils import logging

from ..config import Config


def get_loggers(config: Config):
    # set huggingface verbosity to ERROR
    logging.set_verbosity(verbosity=logging.ERROR)

    loggers = []
    if config.logger.wandb:
        wandb_logger = WandbLogger(
            offline=config.logger.wandb_offline,
            project=config.logger.wandb_project,
            config=OmegaConf.to_container(config, resolve=True, throw_on_missing=True),
        )
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


class ProgressBar(RichProgressBar):
    def __init__(
        self,
        refresh_rate: int = 1,
        leave: bool = False,
        theme: RichProgressBarTheme = RichProgressBarTheme(),
        console_kwargs: Optional[Dict[str, Any]] = None,
    ) -> None:

        super().__init__(refresh_rate, leave, theme, console_kwargs)
