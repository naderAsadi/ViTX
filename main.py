import pytorch_lightning as pl
from pytorch_lightning.strategies import DDPStrategy

from vision_text import (
    config_parser,
    get_dataloaders,
    get_loggers,
    get_method,
    get_model,
)


def main():
    config = config_parser(
        config_path="./configs/", config_name="default", job_name="test"
    )

    train_loader, test_loader = get_dataloaders(
        config=config, return_val_loader=config.train.check_val
    )

    loggers = get_loggers(config=config)

    method = get_method(config=config)

    trainer = pl.Trainer(
        logger=loggers,
        accelerator=config.train.accelerator_type,
        devices=config.train.n_devices,
        strategy=DDPStrategy(find_unused_parameters=False),
        max_epochs=config.train.n_epochs,
        check_val_every_n_epoch=config.train.check_val_every_n_epoch,
    )

    trainer.fit(
        model=method, train_dataloaders=train_loader, val_dataloaders=test_loader
    )
    trainer.test(model=method, dataloaders=test_loader)


if __name__ == "__main__":
    main()
