python main.py method="simclr" \
data="cc3m" data/transform="simclr" data.transform.n_views=2 data.n_workers=16 \
model/vision_model="resnet34" model.vision_model.embed_dim=512 model/optimizer="sgd" \
model.optimizer.scheduler.milestones="30-50-70" \
train.batch_size=512 train.n_epochs=100 train.mixed_precision=True \
logger.log_train_acc=False logger.wandb=True logger.wandb_offline=True