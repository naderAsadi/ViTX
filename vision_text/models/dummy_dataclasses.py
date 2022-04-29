from dataclasses import dataclass
from typing import Any, ContextManager, List, Optional, Tuple

import torch


@dataclass
class ModelOutput:
    pooled_last_hidden: torch.FloatTensor = None
    last_hidden_state: torch.FloatTensor = None
    pooler_output: torch.FloatTensor = None


@dataclass
class VisionTextDualOutput:
    loss: Optional[torch.FloatTensor] = None
    vision_pooled_embeds: torch.FloatTensor = None
    text_pooled_embeds: torch.FloatTensor = None
    vision_model_output: torch.FloatTensor = None
    text_model_output: torch.FloatTensor = None
    logits_per_image: Optional[torch.FloatTensor] = None
    logits_per_text: Optional[torch.FloatTensor] = None
