import torch
from torch.utils.data import DataLoader

from . import get_dataset
from ..config import Config
from ..utils import spinner_animation


@spinner_animation(message="Loading Datasets...")
def get_dataloaders(config: Config):

    data_path = config.data.images_path

    # create train dataset
    config.data.images_path = data_path + "train/"
    train_dataset = get_dataset(data_config=config.data)
    # create test dataset
    config.data.images_path = data_path + "val/"
    test_dataset = get_dataset(data_config=config.data)

    config.data.images_path = data_path

    train_loader = DataLoader(
        dataset=train_dataset,
        batch_size=config.train.batch_size,
        num_workers=config.data.n_workers,
        shuffle=True,
    )
    test_loader = DataLoader(
        dataset=test_dataset,
        batch_size=config.train.batch_size,
        num_workers=config.data.n_workers,
        shuffle=False,
    )

    return train_loader, test_loader
