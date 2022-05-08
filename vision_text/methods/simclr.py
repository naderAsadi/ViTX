from typing import Optional, Union
import torch
import torch.nn as nn


from . import register_method
from .base import BaseMethod
from ..config import Config, OptimizerConfig
from ..losses import SupConLoss
from ..models import MLP, ModelOutput, get_model


@register_method("simclr")
class SimCLR(BaseMethod):
    def __init__(
        self,
        model: nn.Module,
        optim_config: Optional[Union[OptimizerConfig, dict]] = OptimizerConfig(),
    ):
        super(SimCLR, self).__init__(
            model=model,
            optim_config=optim_config,
        )

        self.loss_func = SupConLoss(temperature=0.5)

    @classmethod
    def from_config(cls, config: Config) -> "SimCLR":
        model = get_model(model_config=config.model.vision_model)

        return cls(
            model=model,
            optim_config=config.model.optimizer,
        )

    def forward(
        self,
        pixel_values: torch.FloatTensor,
        return_loss: Optional[bool] = True,
    ):
        assert (
            len(pixel_values.shape) == 5
        ), "`pixel_values` needs to be [bsz, n_views, ...], at least 5 dimensions are required."

        bsz = pixel_values.size(0)
        images = pixel_values.reshape(-1, *pixel_values.shape[2:])

        outputs = self.model(images)
        features = outputs.pooler_output

        features = nn.functional.normalize(features, dim=-1)
        f1, f2 = torch.split(features, [bsz, bsz], dim=0)
        features = torch.cat([f1.unsqueeze(1), f2.unsqueeze(1)], dim=1)

        if return_loss:
            loss = self.loss_func(features=features)
            outputs.loss = loss

        return outputs

    def training_step(self, batch, batch_idx):
        pixel_values = batch[0]
        metrics = {}

        outputs = self.forward(
            pixel_values=pixel_values,
            return_loss=True,
        )

        metrics["train/loss"] = outputs.loss
        self.log_dict(metrics, sync_dist=True)

        return outputs.loss

    def validation_step(self, batch, batch_idx):
        loss = self._shared_eval_step(batch, batch_idx)
        metrics = {"validation/loss": loss}
        self.log_dict(metrics, sync_dist=True)

        return metrics

    def test_step(self, batch, batch_idx):
        loss = self._shared_eval_step(batch, batch_idx)
        metrics = {"test/loss": loss}
        self.log_dict(metrics, sync_dist=True)

        return metrics

    def _shared_eval_step(self, batch, batch_idx):
        pixel_values = batch[0]

        outputs = self.forward(
            pixel_values=pixel_values,
            return_loss=True,
        )

        return outputs.loss

    def predict_step(self, batch, batch_idx):
        pixel_values = batch[0]

        outputs = self.forward(
            pixel_values=pixel_values,
            return_loss=False,
        )

        return outputs
