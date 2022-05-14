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

from ..config import Config, OptimizerConfig


def get_optimizer(parameters, optim_config: OptimizerConfig):

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


def _get_unique_id_from_config(config: Config) -> str:
    temp_config = dict(config)

    r_keys = ["train", "logger"]
    for key in r_keys:
        temp_config.pop(key)

    return hashlib.md5(str(temp_config).encode()).hexdigest()[:8]


def sync_checkpoints(config: Config):
    console = Console()

    if not os.path.exists(config.model.checkpoint_root):
        os.mkdir(config.model.checkpoint_root)

    checkpoint_id = _get_unique_id_from_config(config=config)

    checkpoint_root = Path(config.model.checkpoint_root).joinpath(checkpoint_id)
    config.model.checkpoint_root = str(checkpoint_root.absolute())

    # if no checkpoint is available, create checkpoint dir and save run configs
    if not os.path.exists(config.model.checkpoint_root):
        console.print(
            f"No checkpoints found. Creating checkpoints directory [yellow]`{checkpoint_id}`[/yellow]"
        )
        os.mkdir(config.model.checkpoint_root)

        with open(f"{config.model.checkpoint_root}/config.yaml", "w") as json_file:
            OmegaConf.save(config=config, f=json_file.name)

        return config.model.ckpt_checkpoint_path

    # load ckpt state checkpoints
    checkpoint_files = os.listdir(config.model.checkpoint_root)
    ckpt_files = [f for f in checkpoint_files if "ckpt" in f]
    # return the last ckpt checkpoint
    if len(ckpt_files) > 0:
        console.print(
            f"{len(ckpt_files)} checkpoints found for [yellow]`{checkpoint_id}`[/yellow]. Loading the last ckpt checkpoint: [yellow] {ckpt_files[-1]}[/yellow]",
            highlight=False,
        )
        config.model.ckpt_checkpoint_path = str(
            checkpoint_root.joinpath(ckpt_files[-1]).absolute()
        )

    return config.model.ckpt_checkpoint_path
