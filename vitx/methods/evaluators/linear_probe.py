from typing import Optional, Union

import torch
import torch.nn as nn
import torch.nn.functional as F
import pytorch_lightning as pl

from ...models import VisionTextEncoder


class ProbeEvaluator(pl.LightningModule):
    def __init__(self, model: nn.Module, embed_dim: int, n_classes: int):
        """Linear probe evaluation class.

        Args:
            model  (nn.Module): _description_
            embed_dim (int): _description_
            n_classes (int): _description_
        """
        super(ProbeEvaluator, self).__init__()

        self.model = model
        self.linear_probe = nn.Linear(in_features=embed_dim, out_features=n_classes)

    def configure_optimizers(self):
        return torch.optim.AdamW(
            params=self.linear_probe.parameters(), lr=1e-3, weight_decay=5e-4,
        )

    def forward(
        self,
        pixel_values: torch.FloatTensor,
        target: torch.FloatTensor,
        return_loss: Optional[bool] = True,
    ):
        self.model.eval()
        with torch.inference_mode():
            features = self.model(pixel_values=pixel_values, forward_head=False,)

        logits = self.linear_probe(features.pooler_output.clone())

        loss = None
        if return_loss:
            loss = F.cross_entropy(logits, target)

        return logits, loss

    def training_step(self, batch, batch_idx):
        pixel_values, target = batch
        metrics = {}

        logits, loss = self.forward(pixel_values=pixel_values, target=target)
        acc = torch.sum(logits.argmax(dim=-1) == target) / pixel_values.size(0)

        metrics["train/loss"] = loss
        metrics["train/acc"] = acc

        self.log_dict(metrics, sync_dist=True, prog_bar=True)

        return loss

    def validation_step(self, batch, batch_idx):
        loss = self._shared_eval_step(batch, batch_idx)
        metrics = {"validation/loss": loss}
        self.log_dict(metrics, sync_dist=True, prog_bar=True)

        return metrics

    def test_step(self, batch, batch_idx):
        loss = self._shared_eval_step(batch, batch_idx)
        metrics = {"test/loss": loss}
        self.log_dict(metrics, sync_dist=True, prog_bar=True)

        return metrics

    def _shared_eval_step(self, batch, batch_idx):
        pixel_values, target = batch

        logits, loss = self.forward(
            pixel_values=pixel_values, target=target, return_loss=True,
        )

        return loss
