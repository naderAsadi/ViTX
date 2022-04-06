from typing import Optional

import torch
import torch.nn as nn

from ..config import ModelConfig
from .vision_text_encoder import VisionTextModel


class ModelWrapper(nn.Module):

    def __init__(
        self,
        config: ModelConfig,
        trunk: nn.Module,
        head: Optional[nn.Module] = None
    ):
        super(ModelWrapper, self).__init__()

        self.config = config

        self.trunk = trunk
        self.head = head

    def forward_trunk_vision_model(self, data):
        if not isinstance(self.trunk, VisionTextModel):
            raise ValueError(
                f"trunk model is not an instance of VisionTextModel, use `.forward_trunk` method instead."
            )
        
        return self.trunk.get_image_features(*data)

    def forward_trunk_text_model(self, data):
        if not isinstance(self.trunk, VisionTextModel):
            raise ValueError(
                f"trunk model is not an instance of VisionTextModel, use `.forward_trunk` method instead."
            )
        
        return self.trunk.get_text_features(*data)

    def forward_trunk(self, data):
        return self.trunk(*data)

    def forward_head(self, embeds):
        return self.head(*embeds)

    def forward(self, data):
        embeds = self.trunk(*data)
        return self.head(*embeds)
