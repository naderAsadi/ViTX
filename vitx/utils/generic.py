"""
Generic utilities
"""

import inspect
import hashlib
import os
from omegaconf import OmegaConf
from pathlib import Path
from rich.console import Console

import torch

from ..config import Config, OptimizerConfig, SchedulerConfig


def get_optimizer(parameters, optim_config: OptimizerConfig) -> torch.optim:

    assert hasattr(
        torch.optim, optim_config.name
    ), f"{optim_config.name} is not a registered optimizer in `torch.optim`."

    optim = getattr(torch.optim, optim_config.name)
    optim_args = inspect.getfullargspec(optim).args

    optim_config = dict(optim_config)
    keys = list(optim_config.keys())
    for key in keys:
        if key not in optim_args:
            optim_config.pop(key)

    return optim(parameters, **optim_config)


def get_lr_scheduler(
    optimizer, scheduler_config: SchedulerConfig
) -> torch.optim.lr_scheduler:

    assert hasattr(
        torch.optim.lr_scheduler, scheduler_config.name
    ), f"{scheduler_config.name} is not a registered scheduler in `torch.optim.lr_scheduler`."

    scheduler = getattr(torch.optim.lr_scheduler, scheduler_config.name)
    scheduler_args = inspect.getfullargspec(scheduler).args

    scheduler_config = dict(scheduler_config)

    scheduler_config["milestones"] = list(
        map(int, scheduler_config["milestones"].split("-"))
    )

    keys = list(scheduler_config.keys())
    for key in keys:
        if key not in scheduler_args:
            scheduler_config.pop(key)

    return scheduler(optimizer, **scheduler_config)


def _get_unique_id_from_config(config: Config) -> str:
    config_dict = dict(config)

    config_dict["data"] = {"dataset": config.data.dataset}
    # attributes to discard for the unique hash address
    r_keys = [
        "train",
        "logger",
        "checkpoints_root",
        "unique_run_id",
        "ckpt_checkpoint_path",
    ]
    for key in r_keys:
        config_dict.pop(key)

    return hashlib.md5(str(config_dict).encode()).hexdigest()[:8]


def sync_checkpoints(config: Config):
    console = Console()

    if not os.path.exists(config.checkpoints_root):
        os.mkdir(config.checkpoints_root)

    if config.unique_run_id is None:
        checkpoint_id = _get_unique_id_from_config(config=config)
        config.unique_run_id = checkpoint_id

    checkpoints_root = (
        Path(config.checkpoints_root).joinpath(config.unique_run_id).absolute()
    )

    # if no checkpoint is available, create checkpoint dir and save run configs
    if not os.path.exists(checkpoints_root):
        console.print(
            f"No checkpoints found. Creating checkpoints directory [yellow]`{config.unique_run_id}`[/yellow]"
        )
        os.mkdir(checkpoints_root)

        with open(f"{checkpoints_root}/config.yaml", "w") as json_file:
            OmegaConf.save(config=config, f=json_file.name)

        return config.ckpt_checkpoint_path

    # load ckpt state checkpoints
    checkpoint_files = os.listdir(checkpoints_root)
    ckpt_files = [f for f in checkpoint_files if "ckpt" in f]
    # return the last ckpt checkpoint
    if len(ckpt_files) > 0:
        console.print(
            f"{len(ckpt_files)} checkpoints found for [yellow]`{config.unique_run_id}`[/yellow]. Loading the last ckpt checkpoint: [yellow] {ckpt_files[-1]}[/yellow]",
            highlight=False,
        )
        config.ckpt_checkpoint_path = str(
            checkpoints_root.joinpath(ckpt_files[-1]).absolute()
        )

    return config.ckpt_checkpoint_path
