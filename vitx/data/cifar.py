from typing import Any, Callable, Dict, Optional, Sequence, Union

from torch.utils.data import Dataset
from torchvision.datasets.cifar import CIFAR10, CIFAR100

from . import register_dataset
from .helper import get_image_transforms
from ..config import DataConfig


class CIFARDataset(Dataset):

    _CIFAR_TYPE = None
    _MEAN = (0.4914, 0.4822, 0.4465)
    _STD = (0.2023, 0.1994, 0.2010)
    _IMAGE_SIZE = 32

    def __init__(
        self,
        images_path: str,
        image_transform: Optional = None,
        image_size: Optional[int] = 224,
        train: Optional[bool] = True,
        download: Optional[bool] = True,
    ) -> None:

        assert self._CIFAR_TYPE in [
            "cifar10",
            "cifar100",
        ], "CIFARDataset must be subclassed and a valid _CIFAR_TYPE provided"

        if self._CIFAR_TYPE == "cifar10":
            self.dataset = CIFAR10(images_path, train=train, download=download)
        if self._CIFAR_TYPE == "cifar100":
            self.dataset = CIFAR100(images_path, train=train, download=download)

        if image_transform is None:
            image_transform = T.Compose(
                [
                    T.Resize((image_size, image_size)),
                    T.ToTensor(),
                    T.Normalize(mean=CIFAR100Dataset._MEAN, std=CIFAR100Dataset._STD),
                ]
            )

        self.image_transform = image_transform

    @classmethod
    def from_config(
        cls, data_config: DataConfig, split: str = "train"
    ) -> "CIFARDataset":

        assert split in [
            "train",
            "val",
            "test",
        ], f"`split` should be in [`train`, `val`, `test`], but {split} is entered."

        image_transform = get_image_transforms(
            transform_config=data_config.transform, split=split
        )

        return cls(
            images_path=data_config.train_images_path,
            image_transform=image_transform,
            train=True if split == "train" else False,
            download=True,
        )


@register_dataset("cifar10")
class CIFAR10Dataset(CIFARDataset):
    _CIFAR_TYPE = "cifar10"


@register_dataset("cifar100")
class CIFAR100Dataset(CIFARDataset):
    _CIFAR_TYPE = "cifar100"
