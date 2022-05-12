"""
Generic utilities
"""

import inspect
import torch

from ..config import OptimizerConfig


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
