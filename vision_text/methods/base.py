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
        tokenizer = None,
        feature_extractor = None,
        processor = None,
    ):
        super().__init__()

        self.config = config
        self.trunk = trunk
        self.head = head

    def configure_optimizers(self):
        optimizer = torch.optim.SGD(
            self.trunk.parameters(), 
            lr=self.config.train.lr,
            momentum=self.config.train.momentum,
            weight_decay=self.config.train.weight_decay
        )
        return optimizer

    def _compute_loss(
        self, 
        outputs: Union[VisionOutput, VisionTextOutput]
    ) -> torch.FloatTensor:
        
        raise NotImplementedError
    
    def forward(
        self,
        input_ids,
        attention_mask,
        pixel_values
    ):
        outputs = self.trunk(
            input_ids=input_ids,
            attention_mask=attention_mask,
            pixel_values=pixel_values,
            return_loss=False,
        )

        loss = self._compute_loss(outputs)
        outputs.loss = loss

        return outputs

    def training_step(self, batch, batch_idx):

        raise NotImplementedError

    

            