from typing import Optional, Union

import torch
import torch.nn as nn
import pytorch_lightning as pl

from ..config import Config, OptimizerConfig
from ..models import ModelOutput, VisionTextDualOutput
from ..utils import get_optimizer


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
        optimizers = [
            get_optimizer(self.trunk.parameters(), optim_config=self.trunk_optim_config)
        ]

        if self.head is not None:
            optimizers.append(
                get_optimizer(
                    self.head.parameters(), optim_config=self.head_optim_config
                )
            )

        return optimizers

    def on_before_batch_transfer(self, batch, dataloader_idx):

        super().on_before_batch_transfer(batch, dataloader_idx)

    def forward(
        self,
        pixel_values: torch.FloatTensor,
        input_ids: torch.FloatTensor,
        attention_mask: torch.FloatTensor,
        return_loss: Optional[bool] = True,
    ):
        outputs = self.trunk(
            input_ids=input_ids,
            attention_mask=attention_mask,
            pixel_values=pixel_values,
        )

        if return_loss:
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
