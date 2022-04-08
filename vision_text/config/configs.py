from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple


""" Data Configs """



""" Model Configs """
@dataclass
class VisionModelConfig:
    name: str = "openai/clip-vit-base-patch32" #google/vit-base-patch16-224
    pretrained: bool = True
    embed_dim: int = 768

@dataclass
class TextModelConfig:
    name: str = "openai/clip-vit-base-patch32" #bert-base-uncased
    pretrained: bool = True
    embed_dim: int = 512 #768

@dataclass
class ModelConfig:
    vision_model: VisionModelConfig = VisionModelConfig()
    text_model: TextModelConfig = TextModelConfig()
    projection_dim: int = 512
    logit_scale_init_value: float = 2.6592
    return_dict: bool = True


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
    model: ModelConfig = ModelConfig()
    optim: OptimConfig = OptimConfig()
