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

    @classmethod
    def from_config(cls, config: Config) -> "BaseMethod":

        raise NotImplementedError

    def configure_optimizers(self):
        return get_optimizer(self.model.parameters(), optim_config=self.optim_config)

    def forward(
        self,
        pixel_values: torch.FloatTensor,
        input_ids: torch.FloatTensor,
        attention_mask: torch.FloatTensor,
        return_loss: Optional[bool] = True,
    ):
        outputs = self.model(
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
