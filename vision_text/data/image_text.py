from pathlib import Path
from PIL import Image
from random import randint, choices
from typing import Any, Optional, Tuple

import torch
from torch.utils.data import Dataset
from torchvision import transforms as T

from . import register_dataset
from ..config import DataConfig


@register_dataset("image_text")
class ImageTextDataset(Dataset):
    def __init__(
        self,
        images_path: str,
        ann_file_path: str,
        image_transform: Optional = None,
        image_size: Optional[int] = 224,
        resize_ratio: Optional[float] = 0.75,
        shuffle: Optional[bool] = False,
    ):
        """Create a image-text dataset from a directory with congruent text and image names.

        Args:
            images_path (str): Path to the folder containing images of the dataset.
            ann_file_path (str): Path to the `json` or `csv` annotation file.
            image_transform (_type_, optional): _description_. Defaults to None.
            image_size (Optional[int], optional): The size of outputted images.. Defaults to 224.
            resize_ratio (Optional[float], optional): Minimum percentage of image contained by resize. Defaults to 0.75.
        """
        super(ImageTextDataset, self).__init__()

        # load the captions
        with open(ann_file_path, "r") as file:
            lines = file.readlines()
        captions = {
            line.split("\t", 1)[0]: line.split("\t", 1)[1].strip() for line in lines
        }

        # get image files path
        images_path = Path(images_path)
        image_files = [
            *images_path.glob("**/*.png"),
            *images_path.glob("**/*.jpg"),
            *images_path.glob("**/*.jpeg"),
            *images_path.glob("**/*.bmp"),
        ]
        image_files = {str(file.parts[-1].split(".")[0]): file for file in image_files}

        keys = image_files.keys() & captions.keys()

        self.keys = list(keys)
        self.captions = {k: v for k, v in captions.items() if k in keys}
        self.image_files = {k: v for k, v in image_files.items() if k in keys}

        if image_transform is None:
            image_transform = T.Compose(
                [
                    T.RandomResizedCrop(
                        image_size, scale=(resize_ratio, 1.0), ratio=(1.0, 1.0)
                    ),
                    T.ToTensor(),
                    T.Normalize(
                        mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)
                    ),  # ImageNet mean and std
                ]
            )
        self.image_transform = image_transform
        self.shuffle = shuffle

    @classmethod
    def from_config(
        cls, data_config: DataConfig, split: str = "train"
    ) -> "ImageTextDataset":

        assert split in [
            "train",
            "val",
            "test",
        ], f"`split` should be in [`train`, `val`, `test`], but {split} is entered."

        if split == "train":
            images_path = data_config.train_images_path
            ann_file_path = data_config.train_ann_path
        else:
            images_path = data_config.val_images_path
            ann_file_path = data_config.val_ann_path

        return cls(
            images_path=images_path,
            ann_file_path=ann_file_path,
            image_size=data_config.image_size,
            resize_ratio=data_config.resize_ratio,
        )

    def __len__(self):
        return len(self.keys)

    def random_sample(self):
        return self.__getitem__(randint(0, self.__len__() - 1))

    def sequential_sample(self, idx):
        if idx >= self.__len__() - 1:
            return self.__getitem__(0)
        return self.__getitem__(idx + 1)

    def skip_sample(self, idx):
        if self.shuffle:
            return self.random_sample()
        return self.sequential_sample(idx=idx)

    def __getitem__(self, idx) -> Tuple[Any, Any]:
        key = self.keys[idx]

        caption = self.captions[key]
        try:
            image = self.image_transform(
                Image.open(self.image_files[key]).convert("RGB")
            )
        except:
            return self.skip_sample(idx)

        return image, caption
