from typing import Optional

import torch
import torch.nn as nn

from ..config import ModelConfig



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
