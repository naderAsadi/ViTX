from typing import Optional, Union

import torch
import torch.nn as nn
import pytorch_lightning as pl

from ..config import Config
from ..models import VisionOutput, VisionTextOutput


class BaseMethod(pl.LightningModule):
    def __init__(
        self,
        config: Config,
        trunk: Optional[nn.Module],
        head: Optional[nn.Module] = None,
        tokenizer=None,
    ):
        super().__init__()

        self.config = config
        self.trunk = trunk
        self.head = head
        self.tokenizer = tokenizer

    @classmethod
    def from_config(cls, config: Config):

        raise NotImplementedError

    def configure_optimizers(self):
        params = [{"params": self.trunk.parameters()}]
        if self.head is not None:
            params.append({"params": self.head.parameters()})

        optimizer = torch.optim.SGD(
            params=params,
            lr=self.config.train.lr,
            momentum=self.config.train.momentum,
            weight_decay=self.config.train.weight_decay,
        )
        return optimizer

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
        self, outputs: Union[VisionOutput, VisionTextOutput]
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
