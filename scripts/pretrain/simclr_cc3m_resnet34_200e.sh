python main.py method="simclr" \
data="cc3m" data/transform="simclr" data.transform.n_views=2 data.n_workers=12 \
model/vision_model="resnet34" model.vision_model.embed_dim=512 model/optimizer="sgd" \
model.optimizer.scheduler.milestones="50-100-150" \
train.batch_size=512 train.n_epochs=200 train.mixed_precision=True \
logger.log_train_acc=False logger.wandb=True logger.wandb_offline=True 
data.train_images_path=$SLURM_TMPDIR/CC3M_336/images/train/ \
data.val_images_path=$SLURM_TMPDIR/CC3M_336/images/val/ \
data.train_ann_path=$SLURM_TMPDIR/CC3M_336/annotations/train_captions.tsv \
data.val_ann_path=$SLURM_TMPDIR/CC3M_336/annotations/val_captions.tsv