from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple


@dataclass
class DataConfig:
    dataset: str = "coco"
    images_path: str = "../datasets/coco-caption/images/"
    annotation_path: str = "../datasets/coco-caption/annotations/"
    n_workers: int = 8
    n_frames: int = 8
    # image_transform
    image_size: int = 224
    resize_ratio: float = 0.75


@dataclass
class VisionModelConfig:
    name: str = "openai/clip-vit-base-patch32"  # openai/clip-vit-large-patch14
    pretrained: bool = True
    embed_dim: int = 768


@dataclass
class TextModelConfig:
    name: str = "openai/clip-vit-base-patch32"  # openai/clip-vit-large-patch14
    tokenizer: str = "openai/clip-vit-base-patch32"
    pretrained: bool = True
    embed_dim: int = 512


@dataclass
class ModelConfig:
    vision_model: VisionModelConfig = VisionModelConfig()
    text_model: TextModelConfig = TextModelConfig()
    projection_dim: int = 512
    logit_scale_init_value: float = 2.6592
    checkpoint_root: str = "checkpoints/"


@dataclass
class TrainConfig:
    batch_size: int = 64
    n_epochs: int = 32
    check_val_every_n_epoch: int = 10
    # Optimizer
    optim: str = "sgd"
    lr: float = 5e-3
    momentum: float = 0.9
    weight_decay: float = 1e-4
    # Distributed Training
    accelerator_type: str = "gpu"
    n_devices: int = 1


@dataclass
class LoggerConfig:
    log_train_acc: bool = False
    wandb: bool = False
    wandb_offline: bool = False
    wandb_project: str = "vision-text"


@dataclass
class Config:
    method: str = "clip"
    data: DataConfig = DataConfig()
    model: ModelConfig = ModelConfig()
    train: TrainConfig = TrainConfig()
    logger: LoggerConfig = LoggerConfig()
