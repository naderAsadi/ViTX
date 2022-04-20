import torch
from torch.utils.data import DataLoader
import torchvision.transforms as T
import pytorch_lightning as pl

from transformers import CLIPVisionModel, CLIPTextModel

from vision_text.models import VisionTextModel
from vision_text.config import config_parser
from vision_text.methods import CLIP
from vision_text.data import COCODataset, MPIVideoDataset, get_dataloaders
from vision_text.utils import get_loggers


def main():
    config = config_parser(
        config_path="./configs/", config_name="default", job_name="test"
    )

    # train_loader, test_loader = get_mpi_loaders(config)
    train_loader, test_loader = get_dataloaders(config=config)

    loggers = get_loggers(config=config)

    clip_method = CLIP.from_config(config)

    trainer = pl.Trainer(
        logger=loggers,
        accelerator=config.train.accelerator_type,
        devices=config.train.n_devices,
        max_epochs=config.train.n_epochs,
        check_val_every_n_epoch=config.train.check_val_every_n_epoch,
    )

    trainer.fit(
        model=clip_method, train_dataloaders=train_loader, val_dataloaders=test_loader
    )
    trainer.test(model=clip_method, dataloaders=test_loader)


if __name__ == "__main__":
    main()
