import random

import torch
import pytorch_lightning as pl
from transformers import CLIPTokenizer

from ..config import Config
from ..models import VisionOutput, VisionTextOutput
from ..losses import clip_loss
from .base import BaseMethod


class CLIP(BaseMethod):

    def __init__(
        self, 
        config, 
        trunk
    ):
        super().__init__(config, trunk)

        self.tokenizer = CLIPTokenizer.from_pretrained(self.config.model.text_model.tokenizer)

    def _compute_loss(
        self, 
        outputs: VisionTextOutput
    ) -> torch.FloatTensor:
        
        return clip_loss(outputs.logits_per_text)
    
    def training_step(self, batch, batch_idx):
        images, captions = batch
        device = next(self.trunk.parameters()).device

        text = []
        for cap in captions:
           text.append(random.choice(cap))

        text_inputs = self.tokenizer(text, return_tensors="pt", padding=True)
        pixel_values = torch.stack(images)

        for key, value in text_inputs.items():
                text_inputs[key] = value.to(device)

        outputs = self.trunk(
            input_ids=text_inputs.input_ids,
            attention_mask=text_inputs.attention_mask,
            pixel_values=pixel_values,
            return_loss=False
        )

        loss = self._compute_loss(outputs)
        self.log("train/loss", loss)

        return loss

