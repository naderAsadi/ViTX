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
        trunk: nn.Module,
        head: Optional[nn.Module],
        trunk_optim_config: Optional[Union[OptimizerConfig, dict]] = OptimizerConfig(),
        head_optim_config: Optional[Union[OptimizerConfig, dict]] = OptimizerConfig(),
    ):
        super(SimCLR, self).__init__(
            trunk=trunk,
            head=head,
            trunk_optim_config=trunk_optim_config,
            head_optim_config=head_optim_config,
        )

        self.loss_func = SupConLoss(temperature=0.2)

    @classmethod
    def from_config(cls, config: Config) -> "SimCLR":
        model = get_model(model_config=config.model.vision_model)
        head = MLP(
            input_dim=config.model.vision_model.embed_dim,
            hidden_dim=config.head.hidden_dim,
            output_dim=config.head.output_dim,
            n_layers=config.head.n_layers,
        )
        return cls(
            trunk=model,
            head=head,
            trunk_optim_config=config.model.optimizer,
            head_optim_config=config.head.optimizer,
        )

    def _compute_loss(self, outputs: ModelOutput) -> torch.FloatTensor:

        return self.loss_func(features=outputs.pooler_output)

    def forward(
        self,
        pixel_values: torch.FloatTensor,
        return_loss: Optional[bool] = True,
    ):
        outputs = self.trunk(pixel_values)

        if return_loss:
            loss = self._compute_loss(outputs)
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
