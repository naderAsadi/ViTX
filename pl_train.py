from PIL import Image
import random
import requests

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
import torchvision.transforms as T
import torchvision.datasets as datasets
import pytorch_lightning as pl

from transformers import (
    CLIPVisionModel,
    CLIPTextModel
)

from vision_text.models import VisionTextModel
from vision_text.config import config_parser
from vision_text.methods import CLIP


def collate_fn(batch):
    return tuple(zip(*batch))

def get_train_loader():
    # loading data
    train_data_path = "../datasets/coco-caption/images/train2014/"
    train_ann_path = "../datasets/coco-caption/annotations/captions_train2014.json"

    transform_train = T.Compose(
        [
            T.Resize(224),
            T.RandomCrop((224, 224), padding=4, fill=-1),
            # T.RandomHorizontalFlip(),
            # T.ColorJitter(0.4, 0.4, 0.4),
            T.ToTensor(),
            T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ]
    )
    coco_train_dataset = datasets.CocoCaptions(
        root=train_data_path, annFile=train_ann_path, transform=transform_train
    )
    train_loader = DataLoader(
        coco_train_dataset,
        batch_size=32,
        num_workers=8,
        shuffle=True,
        collate_fn=collate_fn,
    )

    return train_loader


def train():
    config = config_parser(config_path='./configs/', config_name="default", job_name="test")

    vision_model = CLIPVisionModel.from_pretrained(config.model.vision_model.name)
    text_model = CLIPTextModel.from_pretrained(config.model.text_model.name)
    model = VisionTextModel(model_config=config.model, vision_model=vision_model, text_model=text_model)

    train_loader = get_train_loader()
    
    clip_method = CLIP(config, trunk=model)
    trainer = pl.Trainer(accelerator="gpu", devices=1)
    trainer.fit(clip_method, train_loader)


def evaluate():
    model = VisionTextDualEncoderModel.from_pretrained("vit-bert")

    inference
    outputs = model(**inputs)
    logits_per_image = outputs.logits_per_image  # this is the image-text similarity score
    probs = logits_per_image.softmax(dim=1) 


if __name__ == '__main__':
    train()