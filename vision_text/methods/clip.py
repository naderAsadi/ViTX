from typing import Optional
import random

import torch
import pytorch_lightning as pl
from transformers import CLIPTokenizer

from .base import BaseMethod
from ..config import Config
from ..models import VisionTextModel, VisionOutput, VisionTextOutput
from ..losses import clip_loss
from ..utils.metrics import get_retrieval_map, RetrievalMap


class CLIP(BaseMethod):

    EMBED_DIM = 512

    def __init__(
        self,
        config: Config,
        trunk: VisionTextModel,
        head: Optional[torch.nn.Module] = None,
        tokenizer: Optional[CLIPTokenizer] = None,
    ):
        super().__init__(config, trunk=trunk, head=head)

        if tokenizer is None:
            tokenizer = CLIPTokenizer.from_pretrained(
                self.config.model.text_model.tokenizer
            )
        self.tokenizer = tokenizer

    def on_before_batch_transfer(self, batch, dataloader_idx):
        images, captions = batch
        text_inputs = self.tokenizer(captions, return_tensors="pt", padding=True)

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
