from pathlib import Path
import PIL
from random import randint, choice
from typing import Optional

import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms as T
from transformers import CLIPTokenizer

from ..models import VisionTextInput


class ImageTextDataset(Dataset):
    def __init__(
        self,
        root: str,
        tokenizer=None,
        image_transform=None,
        shuffle: Optional[bool] = False,
        image_size: Optional[int] = 224,
        resize_ratio: Optional[float] = 0.75,
        split: Optional[str] = "train",
    ):
        """Create a image-text dataset from a directory with congruent text and image names.

        Args:
            root (str): Folder containing images and text files matched by their paths' respective "stem"
            tokenizer (_type_, optional): Any custom text tokenizer. Defaults to None.
            image_transform (_type_, optional): _description_. Defaults to None.
            shuffle (Optional[bool], optional): _description_. Defaults to False.
            image_size (Optional[int], optional): The size of outputted images.. Defaults to 224.
            resize_ratio (Optional[float], optional): Minimum percentage of image contained by resize. Defaults to 0.75.
            split (Optional[str], optional): Either train, test, or eval. Defaults to 'train'.
        """
        super(ImageTextDataset, self).__init__()

        if split not in ["train", "test", "eval"]:
            raise ValueError(
                f"Data split should be one of [`train`, `test`, `eval`], but was entered: `{split}`"
            )
        if tokenizer is None:
            tokenizer = CLIPTokenizer.from_pretrained("openai/clip-vit-base-patch32")

        self.path = Path(root)
        self.tokenizer = tokenizer

        text_files = [*path.glob("**/*.txt")]
        image_files = [
            *path.glob("**/*.png"),
            *path.glob("**/*.jpg"),
            *path.glob("**/*.jpeg"),
            *path.glob("**/*.bmp"),
        ]

        text_files = {text_file.stem: text_file for text_file in text_files}
        image_files = {image_file.stem: image_file for image_file in image_files}

        keys = image_files.keys() & text_files.keys()

        self.keys = list(keys)
        self.text_files = {k: v for k, v in text_files.items() if k in keys}
        self.image_files = {k: v for k, v in image_files.items() if k in keys}

        if image_transform is None:
            image_transform = T.Compose(
                [
                    T.Lambda(self._convert_to_rgb),
                    T.RandomResizedCrop(
                        image_size, scale=(self.resize_ratio, 1.0), ratio=(1.0, 1.0)
                    ),
                    T.ToTensor(),
                    T.Normalize(
                        mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)
                    ),  # ImageNet mean and std
                ]
            )
        self.image_transform = image_transform

    def __len__(self):
        return len(self.keys)

    def _convert_to_rgb(self, img):
        return img.convert("RGB") if img.mode != "RGB" else img

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

    def __getitem__(self, idx) -> VisionTextInput:
        key = self.keys[idx]

        text_file = self.text_files[key]
        image_file = self.image_files[key]

        descriptions = text_file.read_text().split("\n")
        descriptions = list(filter(lambda t: len(t) > 0, descriptions))
        try:
            description = choice(descriptions)
        except IndexError as zero_captions_in_file_ex:
            print(f"An exception occurred trying to load file {text_file}.")
            print(f"Skipping index {idx}")
            return self.skip_sample(idx)

        tokenized_text = self.tokenizer(description, return_tensors="pt", padding=True)

        try:
            image_tensor = self.image_transform(PIL.Image.open(image_file))
        except (PIL.UnidentifiedImageError, OSError) as corrupt_image_exceptions:
            print(f"An exception occurred trying to load file {image_file}.")
            print(f"Skipping index {idx}")
            return self.skip_sample(idx)

        # Success
        return VisionTextInput(
            pixel_values=image_tensor,
            text_input_ids=tokenized_text.input_ids,
            text_attention_mask=tokenized_text.attention_mask,
        )
