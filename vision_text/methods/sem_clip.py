from typing import Optional, Union
import random

import torch
from torch.nn.functional import softmax

from . import register_method
from .clip import CLIP
from ..losses import clip_loss, cosine_similarity


@register_method("semclip")
class SEMCLIP(CLIP):
    def _compute_loss(self, outputs) -> torch.FloatTensor:

        return clip_loss(outputs.logits_per_text)

    def forward(self, pixel_values, input_ids, attention_mask, return_loss=True):
        outputs = self.trunk(
            input_ids=input_ids,
            attention_mask=attention_mask,
            pixel_values=pixel_values,
            return_similarity_logits=False
        )

        outputs.text_pooled_embeds = softmax(outputs.text_pooled_embeds, dim=-1)
        outputs.vision_pooled_embeds = softmax(outputs.vision_pooled_embeds, dim=-1)

        outputs.logits_per_text, outputs.logits_per_image = cosine_similarity(
            text_embeds=outputs.text_pooled_embeds,
            image_embeds=outputs.vision_pooled_embeds,
            logit_scale=self.trunk.logit_scale,
        )

        if return_loss:
            loss = self._compute_loss(outputs)
            outputs.loss = loss

        return outputs
