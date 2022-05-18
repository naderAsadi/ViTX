import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.strategies import DDPStrategy

from vitx import (
    config_parser,
    get_dataloaders,
    get_loggers,
    get_method,
    get_model,
    sync_checkpoints,
)


def main():
    config = config_parser(
        config_path="./configs/", config_name="default", job_name="test"
    )

    ckpt_checkpoint_path = sync_checkpoints(config=config)

    train_loader, test_loader = get_dataloaders(
        config=config, return_val_loader=config.train.check_val
    )
    method = get_method(config=config)
    loggers = get_loggers(config=config)

    checkpoint_callback = ModelCheckpoint(
        save_top_k=1,
        monitor="epoch",
        mode="max",
        dirpath=config.model.checkpoint_root,
        filename=f"{config.method}-{config.data.dataset}-{config.model.vision_model.name}-"
        + "{epoch:02d}",
    )

    devices = config.train.n_devices
    if config.train.device_ids is not None:
        devices = list(map(int, config.train.device_ids.split("-")))

    trainer = pl.Trainer(
        logger=loggers,
        accelerator=config.train.accelerator_type,
        devices=devices,
        strategy=DDPStrategy(),
        precision=16 if config.train.mixed_precision else 32,
        max_epochs=config.train.n_epochs,
        check_val_every_n_epoch=config.train.check_val_every_n_epoch,
        callbacks=[checkpoint_callback],
    )

    trainer.fit(
        model=method,
        train_dataloaders=train_loader,
        val_dataloaders=test_loader,
        ckpt_path=config.model.ckpt_checkpoint_path,
    )

    trainer.save_checkpoint(
        filepath=f"{config.model.checkpoint_root}/{config.method}-{config.data.dataset}-{config.model.vision_model.name}.pt"
    )


if __name__ == "__main__":
    main()
