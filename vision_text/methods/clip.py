from typing import Optional
import random

import torch
import pytorch_lightning as pl
from transformers import CLIPTokenizer, CLIPVisionModel, CLIPTextModel

from . import register_method
from .base import BaseMethod
from ..config import Config
from ..models import VisionTextModel, VisionOutput, VisionTextOutput
from ..losses import clip_loss
from ..utils.metrics import get_retrieval_map, RetrievalMap


@register_method("clip")
class CLIP(BaseMethod):

    EMBED_DIM = 512

    def __init__(
        self,
        config: Config,
        trunk: VisionTextModel,
        head: Optional[torch.nn.Module] = None,
        tokenizer: Optional[CLIPTokenizer] = None,
    ):

        if tokenizer is None:
            tokenizer = CLIPTokenizer.from_pretrained(
                self.config.model.text_model.tokenizer
            )

        super().__init__(config=config, trunk=trunk, head=head, tokenizer=tokenizer)

    @classmethod
    def from_config(cls, config: Config):
        """Returns an instance of CLIP class from `config` (Config) file.

        Args:
            config (Config): _description_

        Returns:
            _type_: _description_
        """
        vision_model = CLIPVisionModel.from_pretrained(config.model.vision_model.name)
        text_model = CLIPTextModel.from_pretrained(config.model.text_model.name)

        model = VisionTextModel(
            model_config=config.model, vision_model=vision_model, text_model=text_model
        )

        tokenizer = CLIPTokenizer.from_pretrained(config.model.text_model.tokenizer)

        return cls(config=config, trunk=model, tokenizer=tokenizer)

    def on_before_batch_transfer(self, batch, dataloader_idx):
        images, captions = batch
        text_inputs = self.tokenizer(captions, return_tensors="pt", padding=True)

        if text_inputs.input_ids.size(-1) > 77:  # Max sequence length
            text_inputs.input_ids = text_inputs.input_ids[..., :77]
            text_inputs.attention_mask = text_inputs.attention_mask[..., :77]

        return images, text_inputs.input_ids, text_inputs.attention_mask

    def training_step(self, batch, batch_idx):
        pixel_values, text_input_ids, text_attention_mask = batch
        metrics = {}

        outputs = self.trunk(
            input_ids=text_input_ids,
            attention_mask=text_attention_mask,
            pixel_values=pixel_values,
            return_loss=False,
        )
        loss = self._compute_loss(outputs)

        metrics["train/loss"] = loss

        if self.config.logger.log_train_acc:
            retrieval_map = get_retrieval_map(logits_per_text=outputs.logits_per_text)
            acc = retrieval_map.acc_per_text
            metrics["train/acc"] = acc

        self.log_dict(metrics)

        return loss

    def _compute_loss(self, outputs: VisionTextOutput) -> torch.FloatTensor:

        return clip_loss(outputs.logits_per_text)

    def validation_step(self, batch, batch_idx):
        loss, acc = self._shared_eval_step(batch, batch_idx)
        metrics = {"validation/loss": loss, "validation/acc": acc}
        self.log_dict(metrics)

        return metrics

    def test_step(self, batch, batch_idx):
        loss, acc = self._shared_eval_step(batch, batch_idx)
        metrics = {"test/loss": loss, "test/acc": acc}
        self.log_dict(metrics)

        return metrics

    def _shared_eval_step(self, batch, batch_idx):
        pixel_values, text_input_ids, text_attention_mask = batch

        outputs = self.trunk(
            input_ids=text_input_ids,
            attention_mask=text_attention_mask,
            pixel_values=pixel_values,
            return_loss=False,
        )

        loss = self._compute_loss(outputs)
        retrieval_map = get_retrieval_map(logits_per_text=outputs.logits_per_text)
        acc = retrieval_map.acc_per_text

        return loss, acc

    def predict_step(self, batch, batch_idx):
        pixel_values, text_input_ids, text_attention_mask = batch

        outputs = self.trunk(
            input_ids=text_input_ids,
            attention_mask=text_attention_mask,
            pixel_values=pixel_values,
            return_loss=False,
        )

        return outputs
