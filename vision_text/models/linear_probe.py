from typing import Optional, Union

import torch
import torch.nn as nn
import pytorch_lightning as pl

from .vision_text_encoder import VisionTextModel

class ProbeEvaluator(pl.LightningModule):

    def __init__(
        self,
        trunk: Union[VisionTextModel, nn.Module],
        embed_dim: int,
        n_classes: int
    ):
        """Linear probe evaluation class.

        Args:
            trunk (Union[VisionTextModel, nn.Module]): _description_
            embed_dim (int): _description_
            n_classes (int): _description_
        """
        super(ProbeEvaluator, self).__init__()

        self.trunk = trunk
        self.linear_probe = nn.Linear(in_features=embed_dim, out_features=n_classes)

    def forward(self, x):
        self.trunk.eval()
        with torch.inference_mode():
            if 'transformers' in str(type(self.trunk)):
                features = self.trunk(
                    pixel_values=x,
                    output_attentions=None,
                    output_hidden_states=None,
                    return_dict=True,
                )
            else:
                features = self.trunk.return_hidden(x)
        
        logits = self.linear_probe(features)
        return logits