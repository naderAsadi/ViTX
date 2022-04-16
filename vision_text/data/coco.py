import os
import os.path
import random
from PIL import Image
from typing import Any, Callable, Optional, Tuple

from torchvision import transforms as T
import torchvision.datasets as datasets

from ..models import VisionTextInput


class COCODataset(datasets.CocoCaptions):
    def __init__(
        self,
        root: str,
        ann_file_path: str,
        image_transform=None,
        image_size: Optional[int] = 224,
        resize_ratio: Optional[float] = 0.75,
    ):
        """COCO Caption dataset.

        Args:
            root (str): Folder containing images from COCO Caption dataset.
            ann_file_path (str): Path to the COCO Caption annotation file.
            image_transform (_type_, optional): _description_. Defaults to None.
            image_size (Optional[int], optional): The size of outputted images.. Defaults to 224.
            resize_ratio (Optional[float], optional): Minimum percentage of image contained by resize. Defaults to 0.75.
        """

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
        super(COCODataset, self).__init__(
            root=root, annFile=ann_file_path, transform=image_transform
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

        img = Image.open(os.path.join(self.root, path)).convert("RGB")

        if self.transforms is not None:
            img, target = self.transforms(img, target)

        return img, random.choice(target)
