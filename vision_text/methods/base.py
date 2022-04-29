from typing import Optional, Union

import torch
import torch.nn as nn
import pytorch_lightning as pl

from ..config import Config, OptimizerConfig
from ..models import ModelOutput, VisionTextDualOutput


class BaseMethod(pl.LightningModule):
    def __init__(
        self,
        trunk: Optional[nn.Module],
        head: Optional[nn.Module] = None,
        tokenizer: Optional = None,
        max_token_length: Optional[int] = 77,
        trunk_optim_config: Optional[Union[OptimizerConfig, dict]] = OptimizerConfig(),
        head_optim_config: Optional[Union[OptimizerConfig, dict]] = OptimizerConfig(),
        log_train_acc: Optional[bool] = False,
    ):
        super().__init__()

        self.trunk = trunk
        self.head = head
        self.tokenizer = tokenizer

        self.trunk_optim_config = trunk_optim_config
        self.head_optim_config = head_optim_config

        self.max_token_length = max_token_length
        self.log_train_acc = log_train_acc

    @classmethod
    def from_config(cls, config: Config) -> "BaseMethod":

        raise NotImplementedError

    def configure_optimizers(self):
        trunk_optim = torch.optim.SGD(
            params=self.trunk.parameters(),
            lr=self.trunk_optim_config.lr,
            momentum=self.trunk_optim_config.momentum,
            weight_decay=self.trunk_optim_config.weight_decay,
        )
        optimizers = [trunk_optim]

        if self.head is not None:
            head_optim = torch.optim.SGD(
                params=self.head.parameters(),
                lr=self.head_optim_config.lr,
                momentum=self.head_optim_config.momentum,
                weight_decay=self.head_optim_config.weight_decay,
            )

            optimizers.append(head_optim)

        return optimizers

    def on_before_batch_transfer(self, batch, dataloader_idx):

        super().on_before_batch_transfer(batch, dataloader_idx)

    def forward(self, input_ids, attention_mask, pixel_values):
        outputs = self.trunk(
            input_ids=input_ids,
            attention_mask=attention_mask,
            pixel_values=pixel_values,
            return_loss=False,
        )

        loss = self._compute_loss(outputs)
        outputs.loss = loss

        return outputs

    def _compute_loss(
        self, outputs: Union[ModelOutput, VisionTextDualOutput]
    ) -> torch.FloatTensor:

        raise NotImplementedError

    def training_step(self, batch, batch_idx):

        raise NotImplementedError

    def validation_step(self, batch, batch_idx):

        raise NotImplementedError

    def test_step(self, batch, batch_idx):

        raise NotImplementedError

    def predict_step(self, batch, batch_idx):

        raise NotImplementedError
