import os
import hashlib
from pathlib import Path
from omegaconf import OmegaConf

import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.strategies import DDPStrategy

from vision_text import (
    config_parser,
    get_dataloaders,
    get_loggers,
    get_method,
    get_model,
)


def sync_checkpoints(config):
    if not os.path.exists(config.model.checkpoint_root):
        os.mkdir(config.model.checkpoint_root)

    checkpoint_id = hashlib.md5(str(config).encode()).hexdigest()[:8]
    checkpoint_root = Path(config.model.checkpoint_root).joinpath(checkpoint_id)
    config.model.checkpoint_root = str(checkpoint_root.absolute())

    if not os.path.exists(config.model.checkpoint_root):
        os.mkdir(config.model.checkpoint_root)
        with open(f"{config.model.checkpoint_root}/config.yaml", "w") as json_file:
            OmegaConf.save(config=config, f=json_file.name)


def main():
    config = config_parser(
        config_path="./configs/", config_name="default", job_name="test"
    )

    sync_checkpoints(config=config)

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

    trainer = pl.Trainer(
        logger=loggers,
        accelerator=config.train.accelerator_type,
        devices=config.train.n_devices,
        strategy=DDPStrategy(find_unused_parameters=False),
        max_epochs=config.train.n_epochs,
        check_val_every_n_epoch=config.train.check_val_every_n_epoch,
        callbacks=[checkpoint_callback],
    )

    trainer.fit(
        model=method, train_dataloaders=train_loader, val_dataloaders=test_loader
    )

    trainer.save_checkpoint(
        filepath=f"{config.model.checkpoint_root}/{config.method}-{config.data.dataset}-{config.model.vision_model.name}.pt"
    )


if __name__ == "__main__":
    main()
