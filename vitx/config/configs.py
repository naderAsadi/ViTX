import dataclasses
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple


@dataclass
class TransformConfig:
    image_size: int = 224
    resize_ratio: float = 0.75
    horizontal_flip: float = 0.5
    color_jitter: float = 0.4
    random_grayscale: float = 0.2
    n_views: int = 1


@dataclass
class DataConfig:
    dataset: str = "coco"
    train_images_path: str = "../datasets/coco-caption/images/train/"
    val_images_path: str = "../datasets/coco-caption/images/val/"
    train_ann_path: str = "../datasets/coco-caption/annotations/captions_train2014.json"
    val_ann_path: str = "../datasets/coco-caption/annotations/captions_val2014.json"
    n_workers: int = 10
    n_frames: int = 8
    transform: TransformConfig = TransformConfig()


@dataclass
class SchedulerConfig:
    name: str = "MultiStepLR"
    milestones: str = "30-50-70"
    gamma: float = 0.1
    last_epoch: int = -1
    verbose: bool = True


@dataclass
class OptimizerConfig:
    name: str = "SGD"
    lr: float = 0.1
    momentum: float = 0.9
    weight_decay: float = 1e-4
    beta1: float = 0.9
    beta2: float = 0.999
    eps: float = 1e-8
    scheduler: SchedulerConfig = SchedulerConfig()


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
    vocab_size: int = 49408
    embed_dim: int = 512
    n_hidden_layers: int = 12
    n_attention_heads: int = 8
    max_token_length: int = 77  # default CLIP token length


@dataclass
class ModelConfig:
    vision_model: VisionModelConfig = VisionModelConfig()
    text_model: TextModelConfig = TextModelConfig()
    optimizer: OptimizerConfig = OptimizerConfig()
    projection_dim: int = 128
    logit_scale_init_value: float = 2.6592
    temperature: float = 0.1
    topk: int = 1


@dataclass
class TrainConfig:
    batch_size: int = 128
    n_epochs: int = 200
    check_val: bool = True
    check_val_every_n_epoch: int = 5
    # Distributed Training
    accelerator_type: str = "gpu"
    n_devices: int = -1
    device_ids: Optional[str] = None
    mixed_precision: bool = False


@dataclass
class LoggerConfig:
    log_train_acc: bool = False
    wandb: bool = False
    wandb_offline: bool = False
    wandb_project: str = "ViTX"


@dataclass
class Config:
    data: DataConfig = DataConfig()
    model: ModelConfig = ModelConfig()
    train: TrainConfig = TrainConfig()
    logger: LoggerConfig = LoggerConfig()
    method: str = "clip"
    checkpoints_root: str = "./checkpoints/"
    unique_run_id: Optional[str] = None
    ckpt_checkpoint_path: Optional[str] = None
