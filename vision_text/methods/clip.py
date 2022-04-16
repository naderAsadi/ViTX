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
    ):
        super().__init__(config, trunk=trunk, head=head)

        self.tokenizer = CLIPTokenizer.from_pretrained(
            self.config.model.text_model.tokenizer
        )

    def _compute_loss(self, outputs: VisionTextOutput) -> torch.FloatTensor:

        return clip_loss(outputs.logits_per_text)

    def on_before_batch_transfer(self, batch, dataloader_idx):
        image, caption = batch
        text_inputs = self.tokenizer(caption, return_tensors="pt", padding=True)

        return image, text_inputs.input_ids, text_inputs.attention_mask

    def training_step(self, batch, batch_idx):
        images, text_input_ids, text_attention_mask = batch
        device = next(self.trunk.parameters()).device

        # print(images.shape, text_input_ids.shape, text_attention_mask.shape)

        # text = []
        # for cap in captions:
        #     text.append(random.choice(cap))

        # text_inputs = self.tokenizer(text, return_tensors="pt", padding=True)
        # pixel_values = torch.stack(images)

        # for key, value in text_inputs.items():
        #     text_inputs[key] = value.to(device)

        outputs = self.trunk(
            input_ids=text_input_ids,
            attention_mask=text_attention_mask,
            pixel_values=images,
            return_loss=False,
        )

        loss = self._compute_loss(outputs)
        self.log("train/loss", loss)

        return loss

    # def validation_step(self, batch, batch_idx):

    # def test_step(self, batch, batch_idx):

    # def _shared_eval_step(self, batch, batch_idx):
    #     pass

    # def predict_step(self, batch, batch_idx):
        
