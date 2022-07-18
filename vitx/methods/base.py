from typing import Optional, Union

import torch
import torch.nn as nn
import pytorch_lightning as pl

from ..config import Config, OptimizerConfig, SchedulerConfig
from ..models import ModelOutput, VisionTextDualOutput
from ..utils import get_optimizer, get_lr_scheduler


class BaseMethod(pl.LightningModule):
    def __init__(
        self,
        model: Optional[nn.Module],
        tokenizer: Optional = None,
        max_token_length: Optional[int] = 77,
        optim_config: Optional[Union[OptimizerConfig, dict]] = OptimizerConfig(),
        log_train_acc: Optional[bool] = False,
    ):
        super().__init__()

        self.model = model
        self.tokenizer = tokenizer

        self.optim_config = optim_config
        self.max_token_length = max_token_length
        self.log_train_acc = log_train_acc

    @property
    def model_parameters(self):

        return self.model.parameters()

    @classmethod
    def from_config(cls, config: Config) -> "BaseMethod":

        pass

    def training_step(self, batch, batch_idx):

        raise NotImplementedError

    def validation_step(self, batch, batch_idx):

        raise NotImplementedError

    def test_step(self, batch, batch_idx):

        raise NotImplementedError

    def predict_step(self, batch, batch_idx):

        raise NotImplementedError

    def configure_optimizers(self):
        optimizer = get_optimizer(self.model_parameters, optim_config=self.optim_config)

        lr_scheduler = {
            "scheduler": get_lr_scheduler(
                optimizer, scheduler_config=self.optim_config.scheduler
            )
        }

        return {"optimizer": optimizer, "lr_scheduler": lr_scheduler}
