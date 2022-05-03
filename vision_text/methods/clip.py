from typing import Optional, Union

import torch
from transformers import CLIPTokenizer

from . import register_method
from .base import BaseMethod
from ..config import Config, OptimizerConfig
from ..models import VisionTextEncoder, VisionTextDualOutput, get_model
from ..losses import clip_loss
from ..utils.metrics import get_retrieval_map, RetrievalMap


@register_method("clip")
class CLIP(BaseMethod):
    def __init__(
        self,
        trunk: VisionTextEncoder,
        tokenizer: CLIPTokenizer,
        max_token_length: Optional[int] = 77,
        trunk_optim_config: Optional[Union[OptimizerConfig, dict]] = OptimizerConfig(),
        head_optim_config: Optional[Union[OptimizerConfig, dict]] = OptimizerConfig(),
        log_train_acc: Optional[bool] = False,
    ):
        super().__init__(
            trunk=trunk,
            tokenizer=tokenizer,
            max_token_length=max_token_length,
            trunk_optim_config=trunk_optim_config,
            head_optim_config=head_optim_config,
            log_train_acc=log_train_acc,
        )

    @classmethod
    def from_config(cls, config: Config) -> "CLIP":
        """Returns an instance of CLIP class from `config` (Config) file.

        Args:
            config (Config): root config file.

        Returns:
            CLIP: an instance of CLIP method class.
        """
        vision_model = get_model(model_config=config.model.vision_model)
        text_model = get_model(model_config=config.model.text_model)

        trunk_model = VisionTextEncoder(
            vision_model=vision_model,
            text_model=text_model,
            vision_embed_dim=config.model.vision_model.embed_dim,
            text_embed_dim=config.model.text_model.embed_dim,
            projection_dim=config.model.projection_dim,
            logit_scale_init_value=config.model.logit_scale_init_value,
        )

        tokenizer = CLIPTokenizer.from_pretrained(config.model.text_model.tokenizer)

        return cls(
            trunk=trunk_model,
            tokenizer=tokenizer,
            max_token_length=config.model.text_model.max_token_length,
            trunk_optim_config=config.model.optimizer,
            head_optim_config=config.head.optimizer,
            log_train_acc=config.logger.log_train_acc,
        )

    def on_before_batch_transfer(self, batch, dataloader_idx):
        images, captions = batch
        text_inputs = self.tokenizer(
            captions,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=self.max_token_length,
        )

        return images, text_inputs.input_ids, text_inputs.attention_mask

    def training_step(self, batch, batch_idx):
        pixel_values, text_input_ids, text_attention_mask = batch
        metrics = {}

        outputs = self.forward(
            pixel_values=pixel_values,
            input_ids=text_input_ids,
            attention_mask=text_attention_mask,
            return_loss=True,
        )

        metrics["train/loss"] = outputs.loss

        if self.log_train_acc:
            retrieval_map = get_retrieval_map(logits_per_text=outputs.logits_per_text)
            acc = retrieval_map.acc_per_text
            metrics["train/acc"] = acc

        self.log_dict(metrics, sync_dist=True)

        return outputs.loss

    def _compute_loss(self, outputs: VisionTextDualOutput) -> torch.FloatTensor:

        return clip_loss(outputs.logits_per_text)

    def validation_step(self, batch, batch_idx):
        loss, acc = self._shared_eval_step(batch, batch_idx)
        metrics = {"validation/loss": loss, "validation/acc": acc}
        self.log_dict(metrics, sync_dist=True)

        return metrics

    def test_step(self, batch, batch_idx):
        loss, acc = self._shared_eval_step(batch, batch_idx)
        metrics = {"test/loss": loss, "test/acc": acc}
        self.log_dict(metrics, sync_dist=True)

        return metrics

    def _shared_eval_step(self, batch, batch_idx):
        pixel_values, text_input_ids, text_attention_mask = batch

        outputs = self.forward(
            pixel_values=pixel_values,
            input_ids=text_input_ids,
            attention_mask=text_attention_mask,
            return_loss=True,
        )

        retrieval_map = get_retrieval_map(logits_per_text=outputs.logits_per_text)
        acc = retrieval_map.acc_per_text

        return outputs.loss, acc

    def predict_step(self, batch, batch_idx):
        pixel_values, text_input_ids, text_attention_mask = batch

        outputs = self.forward(
            pixel_values=pixel_values,
            input_ids=text_input_ids,
            attention_mask=text_attention_mask,
            return_loss=False,
        )

        return outputs
