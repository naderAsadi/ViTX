from typing import Optional, Union

import torch
import torch.nn as nn
import pytorch_lightning as pl

from ...models import VisionTextEncoder


class ProbeEvaluator(pl.LightningModule):
    def __init__(
        self, model: Union[VisionTextEncoder, nn.Module], embed_dim: int, n_classes: int
    ):
        """Linear probe evaluation class.

        Args:
            model (Union[VisionTextEncoder, nn.Module]): _description_
            embed_dim (int): _description_
            n_classes (int): _description_
        """
        super(ProbeEvaluator, self).__init__()

        self.model = model
        self.linear_probe = nn.Linear(in_features=embed_dim, out_features=n_classes)

    def forward(self, x):
        self.model.eval()
        with torch.inference_mode():
            if "transformers" in str(type(self.model)):
                features = self.model(
                    pixel_values=x,
                    output_attentions=None,
                    output_hidden_states=None,
                    return_dict=True,
                )
            else:
                features = self.model.return_hidden(x)

        logits = self.linear_probe(features)
        return logits
