import torch
from torch.utils.data import DataLoader
import torchvision.transforms as T
import pytorch_lightning as pl
from pytorch_lightning.loggers import WandbLogger

from transformers import CLIPVisionModel, CLIPTextModel

from vision_text.models import VisionTextModel
from vision_text.config import config_parser
from vision_text.methods import CLIP
from vision_text.data import COCODataset


def get_train_loader(config):
    train_data_path = config.data.path + "/images/train2014/"
    train_ann_path = config.data.path + "/annotations/captions_train2014.json"
    test_data_path = config.data.path + "/images/val2014/"
    test_ann_path = config.data.path + "/annotations/captions_val2014.json"

    transform_train = T.Compose(
        [
            T.Resize(224),
            T.RandomCrop((224, 224), padding=4, fill=-1),
            T.ToTensor(),
            T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ]
    )

    coco_train_dataset = COCODataset(
        root=train_data_path,
        ann_file_path=train_ann_path,
        image_transform=transform_train,
    )
    coco_test_dataset = COCODataset(
        root=test_data_path,
        ann_file_path=test_ann_path,
        image_transform=transform_train,
    )

    train_loader = DataLoader(
        coco_train_dataset,
        batch_size=config.train.batch_size,
        num_workers=config.data.n_workers,
        shuffle=True,
    )
    test_loader = DataLoader(
        coco_test_dataset,
        batch_size=config.train.batch_size,
        num_workers=config.data.n_workers,
        shuffle=False,
    )

    return train_loader, test_loader


def train():
    config = config_parser(
        config_path="./configs/", config_name="default", job_name="test"
    )

    vision_model = CLIPVisionModel.from_pretrained(config.model.vision_model.name)
    text_model = CLIPTextModel.from_pretrained(config.model.text_model.name)
    model = VisionTextModel(
        model_config=config.model, vision_model=vision_model, text_model=text_model
    )

    train_loader, test_loader = get_train_loader(config)
    clip_method = CLIP(config, trunk=model)

    loggers = []
    if config.logger.wandb:
        wandb_logger = WandbLogger(
            offline=config.logger.wandb_offline, project=config.logger.wandb_project
        )
        wandb_logger.experiment.config.update(config)
        loggers.append(wandb_logger)

    trainer = pl.Trainer(
        logger=loggers,
        accelerator=config.train.accelerator_type,
        devices=config.train.n_devices,
        max_epochs=config.train.n_epochs,
    )

    trainer.fit(model=clip_method, train_dataloaders=train_loader)
    trainer.test(model=clip_method, dataloaders=test_loader)


if __name__ == "__main__":
    train()
