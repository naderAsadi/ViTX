from typing import Optional
import torch
from torch.utils.data import DataLoader
from torchvision import transforms as T

from . import get_dataset
from .collate import ImageCollateFunction, MultiViewCollateFunction
from ..config import Config, TransformConfig
from ..utils import spinner_animation


@spinner_animation(message="Loading Datasets...")
def get_dataloaders(config: Config, return_val_loader: Optional[bool] = True):

    train_dataset = get_dataset(
        data_config=config.data, split="train", no_transform=True
    )

    train_transform = get_image_transforms(
        transform_config=config.data.transform, split="train"
    )
    if config.data.transform.n_views == 1:
        train_collate_fn = ImageCollateFunction(transform=train_transform)
    else:
        train_collate_fn = MultiViewCollateFunction(transform=train_transform)

    train_loader = DataLoader(
        dataset=train_dataset,
        batch_size=config.train.batch_size,
        num_workers=config.data.n_workers,
        shuffle=True,
        collate_fn=train_collate_fn,
    )

    if return_val_loader:
        val_dataset = get_dataset(data_config=config.data, split="val")

        val_loader = DataLoader(
            dataset=val_dataset,
            batch_size=config.train.batch_size,
            num_workers=config.data.n_workers,
            shuffle=False,
        )

        return train_loader, val_loader

    return train_loader


def get_image_transforms(
    transform_config: TransformConfig, split: Optional[str] = None
) -> T.Compose:
    """_summary_

    Args:
        transform_config (TransformConfig): _description_
        split (Optional[str], optional): _description_. Defaults to None.

    Returns:
        torchvision.transforms.Compose: _description_
    """

    if split in ["val", "test"]:
        return T.Compose(
            [
                T.Resize((transform_config.image_size, transform_config.image_size)),
                T.ToTensor(),
                T.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
            ]
        )

    image_transforms = [
        T.RandomResizedCrop(
            transform_config.image_size,
            scale=(transform_config.resize_ratio, 1.0),
            ratio=(1.0, 1.0),
        )
    ]

    if transform_config.horizontal_flip > 0.0:
        image_transforms.append(
            T.RandomHorizontalFlip(p=transform_config.horizontal_flip)
        )

    if transform_config.color_jitter > 0.0:
        image_transforms.append(
            T.RandomApply(
                [
                    T.ColorJitter(
                        transform_config.color_jitter,
                        transform_config.color_jitter,
                        transform_config.color_jitter,
                        0.1,
                    )
                ],
                p=0.8,
            )
        )

    if transform_config.random_grayscale > 0.0:
        image_transforms.append(T.RandomGrayscale(p=transform_config.random_grayscale))

    image_transforms += [
        T.ToTensor(),
        T.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
    ]

    return T.Compose(image_transforms)
