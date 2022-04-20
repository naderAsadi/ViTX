from pathlib import Path
from PIL import Image
from random import randint, choices
from typing import Any, Optional, Tuple

import torch
from torch.utils.data import Dataset
from torchvision import transforms as T

from . import register_dataset
from ..config import DataConfig


@register_dataset("mpi")
class MPIVideoDataset(Dataset):
    def __init__(
        self,
        images_path: str,
        ann_file_path: str,
        n_frames: Optional[int] = 8,
        image_transform: Optional = None,
        image_size: Optional[int] = 224,
        resize_ratio: Optional[float] = 0.75,
    ):
        """Create a image-text dataset from a directory with congruent text and image names.

        Args:
            images_path (str): Path to the folder containing images of the dataset.
            ann_file_path (str): Path to the `json` or `csv` annotation file.
            image_transform (_type_, optional): _description_. Defaults to None.
            image_size (Optional[int], optional): The size of outputted images.. Defaults to 224.
            resize_ratio (Optional[float], optional): Minimum percentage of image contained by resize. Defaults to 0.75.
        """
        super(MPIVideoDataset, self).__init__()

        # load the captions
        with open(ann_file_path, "r") as file:
            lines = file.readlines()
        captions = {
            line.split("\t", 1)[0]: line.split("\t", 1)[1].strip() for line in lines
        }

        # get image files path
        images_path = Path(images_path)
        all_image_files = [
            *images_path.glob("**/*.png"),
            *images_path.glob("**/*.jpg"),
            *images_path.glob("**/*.jpeg"),
            *images_path.glob("**/*.bmp"),
        ]

        image_files = {}
        for i in range(len(all_image_files)):
            key = all_image_files[i].parts[-2]
            if key in image_files.keys():
                image_files[key].append(all_image_files[i])
            else:
                image_files[key] = [all_image_files[i]]

        keys = image_files.keys() & captions.keys()

        self.n_frames = n_frames
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

    @classmethod
    def from_config(cls, config: DataConfig) -> "MPIVideoDataset":

        return cls(
            images_path=config.images_path,
            ann_file_path=config.annotation_path,
            n_frames=config.n_frames,
            image_size=config.image_size,
            resize_ratio=config.resize_ratio,
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
        image_files = choices(
            self.image_files[key], k=min(self.n_frames, len(self.image_files[key]))
        )

        image_tensors = []
        for image_file in image_files:
            img = Image.open(image_file).convert("RGB")
            image_tensors.append(self.image_transform(img).unsqueeze(1))

        return torch.stack(image_tensors, dim=1).squeeze(), caption
