import os
import os.path
import random
from PIL import Image
from typing import Any, Callable, Optional, Tuple

import torch
import torchvision
import torchvision.datasets as datasets

from . import register_dataset
from .utils import get_image_transforms
from ..config import DataConfig


@register_dataset("coco")
class COCODataset(datasets.CocoCaptions):
    def __init__(
        self,
        images_path: str,
        ann_file_path: str,
        image_transform: torchvision.transforms.Compose,
    ):
        """_summary_

        Args:
            images_path (str): _description_
            ann_file_path (str): _description_
            image_transform (torchvision.transforms.Compose): _description_
        """

        super(COCODataset, self).__init__(
            root=images_path, annFile=ann_file_path, transform=image_transform
        )

    @classmethod
    def from_config(
        cls, data_config: DataConfig, split: str = "train", no_transform: bool = False
    ) -> "COCODataset":

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

        image_transform = None
        if not no_transform:
            image_transform = get_image_transforms(
                transform_config=data_config.transform, split=split
            )

        return cls(
            images_path=images_path,
            ann_file_path=ann_file_path,
            image_transform=image_transform,
        )

    def __getitem__(self, index: int) -> Tuple[Any, Any]:
        """
        Args:
            index (int): Index

        Returns:
            tuple: Tuple (image, target). target is a list of captions for the image.
        """
        coco = self.coco
        img_id = self.ids[index]
        ann_ids = coco.getAnnIds(imgIds=img_id)
        anns = coco.loadAnns(ann_ids)
        target = [ann["caption"] for ann in anns]

        path = coco.loadImgs(img_id)[0]["file_name"]

        image = Image.open(os.path.join(self.root, path)).convert("RGB")

        if self.transforms is not None:
            image, target = self.transforms(image, target)

        return image, random.choice(target)
