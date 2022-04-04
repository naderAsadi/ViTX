from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple


""" Data Configs """



""" Model Configs """

@dataclass
class VisionModelConfig:
    name: str = "resnet50"
    pretrained: bool = True
    projection_size: int = 512

""" Method Configs """

@dataclass
class OptimConfig:
    name: str = "sgd"
    lr: float = 0.01
    momentum: float = 0.9
    weight_decay: float = 0.0001


""" Root Config """
@dataclass
class Config:
    vision_model: VisionModelConfig = VisionModelConfig()
    optim: OptimConfig = OptimConfig()
