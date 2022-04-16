from pytorch_lightning.loggers import WandbLogger

from ..config import LoggerConfig


def get_loggers(logger_config: LoggerConfig):
    loggers = []
    if logger_config.wandb:
        wandb_logger = WandbLogger(
            offline=logger_config.wandb_offline, project=logger_config.wandb_project
        )
        wandb_logger.experiment.config.update(config)
        loggers.append(wandb_logger)

    return loggers
