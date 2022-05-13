from typing import Optional, Union
import torch
import torch.nn as nn


from . import register_method
from .base import BaseMethod
from ..config import Config, OptimizerConfig
from ..losses import supcon_loss
from ..models import MLP, ModelOutput, get_model
from ..utils import get_optimizer


@register_method("simclr")
class SimCLR(BaseMethod):
    def __init__(
        self,
        model: nn.Module,
        projection_head: nn.Module,
        temperature: Optional[float] = 0.3,
        optim_config: Optional[Union[OptimizerConfig, dict]] = OptimizerConfig(),
    ):
        super(SimCLR, self).__init__(
            model=model,
            optim_config=optim_config,
        )
        self.projection_head = projection_head
        self.temperature = temperature

    @classmethod
    def from_config(cls, config: Config) -> "SimCLR":
        model = get_model(model_config=config.model.vision_model)
        projection_head = MLP(
            input_dim=config.model.vision_model.embed_dim,
            hidden_dim=config.model.vision_model.embed_dim,
            output_dim=config.model.projection_dim,
            n_layers=2,
        )

        return cls(
            model=model,
            projection_head=projection_head,
            temperature=config.model.temperature,
            optim_config=config.model.optimizer,
        )

    def configure_optimizers(self):
        return get_optimizer(
            list(self.model.parameters()) + list(self.projection_head.parameters()),
            optim_config=self.optim_config,
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
        image_views = torch.split(pixel_values, split_size_or_sections=1, dim=1)
        pixel_values = torch.cat(list(image_views), dim=0).squeeze()

        outputs = self.model(pixel_values=pixel_values, forward_head=False)
        features = self.projection_head(outputs.pooler_output)

        features = nn.functional.normalize(features, dim=-1)
        f1, f2 = torch.split(features, [bsz, bsz], dim=0)
        features = torch.cat([f1.unsqueeze(1), f2.unsqueeze(1)], dim=1)

        if return_loss:
            loss = supcon_loss(features=features, temperature=self.temperature)
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
