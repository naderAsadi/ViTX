import torch
from torch.utils.data import DataLoader

from . import get_dataset
from ..config import Config
from ..utils import spinner_animation


@spinner_animation(message="Loading Datasets...")
def get_dataloaders(config: Config):

    train_dataset = get_dataset(data_config=config.data, split="train")
    val_dataset = get_dataset(data_config=config.data, split="val")

    train_loader = DataLoader(
        dataset=train_dataset,
        batch_size=config.train.batch_size,
        num_workers=config.data.n_workers,
        shuffle=True,
    )
    val_loader = DataLoader(
        dataset=val_dataset,
        batch_size=config.train.batch_size,
        num_workers=config.data.n_workers,
        shuffle=False,
    )

    return train_loader, val_loader
