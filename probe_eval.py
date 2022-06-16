import torch
from torch.utils.data import DataLoader
import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.strategies import DDPStrategy

from vitx import config_parser, get_method, sync_checkpoints
from vitx.data import CIFAR100Dataset, get_image_transforms
from vitx.methods.evaluators import ProbeEvaluator


def main():
    config = config_parser(
        config_path="./configs/", config_name="default", job_name="test"
    )

    ckpt_checkpoint_path = sync_checkpoints(config=config)

    image_transform = get_image_transforms(transform_config=config.data.transform)
    train_dataset = CIFAR100Dataset(
        images_path="../datasets/cl-datasets/data/",
        image_transform=image_transform,
        train=True,
    )
    test_dataset = CIFAR100Dataset(
        images_path="../datasets/cl-datasets/data/",
        image_transform=image_transform,
        train=False,
    )
    train_loader = DataLoader(
        dataset=train_dataset,
        batch_size=config.train.batch_size,
        num_workers=config.data.n_workers,
        shuffle=True,
    )
    test_loader = DataLoader(
        dataset=test_dataset,
        batch_size=config.train.batch_size,
        num_workers=config.data.n_workers,
        shuffle=False,
    )

    checkpoint = torch.load(config.ckpt_checkpoint_path)
    method = get_method(config=config)
    method.load_state_dict(checkpoint["state_dict"])

    probe_evaluator = ProbeEvaluator(
        model=method.model, embed_dim=config.model.vision_model.embed_dim, n_classes=100
    )

    trainer = pl.Trainer(
        accelerator=config.train.accelerator_type,
        devices=config.train.n_devices,
        strategy=DDPStrategy(),
        precision=16 if config.train.mixed_precision else 32,
        max_epochs=config.train.n_epochs,
        check_val_every_n_epoch=config.train.check_val_every_n_epoch,
    )

    trainer.fit(
        model=probe_evaluator,
        train_dataloaders=train_loader,
        val_dataloaders=test_loader,
    )


if __name__ == "__main__":
    main()
