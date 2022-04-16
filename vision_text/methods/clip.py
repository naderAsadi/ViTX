from typing import Optional
import random

import torch
import pytorch_lightning as pl
from transformers import CLIPTokenizer

from ..config import Config
from ..models import VisionTextModel, VisionOutput, VisionTextOutput
from ..losses import clip_loss
from .base import BaseMethod


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

    def _compute_loss(self, outputs: VisionTextOutput) -> torch.FloatTensor:

        return clip_loss(outputs.logits_per_text)

    def on_before_batch_transfer(self, batch, dataloader_idx):
        images, captions = batch
        text_inputs = self.tokenizer(captions, return_tensors="pt", padding=True)

        return images, text_inputs.input_ids, text_inputs.attention_mask

    def training_step(self, batch, batch_idx):
        pixel_values, text_input_ids, text_attention_mask = batch

        outputs = self.trunk(
            input_ids=text_input_ids,
            attention_mask=text_attention_mask,
            pixel_values=pixel_values,
            return_loss=False,
        )

        loss = self._compute_loss(outputs)
        self.log("train/loss", loss)

        return loss

    # def validation_step(self, batch, batch_idx):

    # def test_step(self, batch, batch_idx):

    # def _shared_eval_step(self, batch, batch_idx):

    # def predict_step(self, batch, batch_idx):
