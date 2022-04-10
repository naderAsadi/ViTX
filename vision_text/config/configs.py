from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple


""" Data Configs """
@dataclass
class DataConfig:
    dataset: str = "coco"
    path: str = "../datasets/coco-caption"
    n_workers: int = 8


""" Model Configs """
@dataclass
class VisionModelConfig:
    name: str = "openai/clip-vit-base-patch32" #google/vit-base-patch16-224
    pretrained: bool = True
    embed_dim: int = 768


@dataclass
class TextModelConfig:
    name: str = "openai/clip-vit-base-patch32" #bert-base-uncased
    tokenizer: str = 'openai/clip-vit-base-patch32'
    pretrained: bool = True
    embed_dim: int = 512 #768


@dataclass
class ModelConfig:
    vision_model: VisionModelConfig = VisionModelConfig()
    text_model: TextModelConfig = TextModelConfig()
    projection_dim: int = 512
    logit_scale_init_value: float = 2.6592
    return_dict: bool = True


@dataclass
class TrainConfig:
    batch_size: int = 32
    n_epochs: int = 32
    optim: str = "sgd"
    lr: float = 0.01
    momentum: float = 0.9
    weight_decay: float = 0.0001


""" Root Config """
@dataclass
class Config:
    data: DataConfig = DataConfig()
    model: ModelConfig = ModelConfig()
    train: TrainConfig = TrainConfig()
