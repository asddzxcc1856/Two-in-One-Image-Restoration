import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader

from utils.dataset_utils import TrainDataset, ValDataset
from net.model_AdaIR import AdaIR
from utils.schedulers import LinearWarmupCosineAnnealingLR
from options import options as opt
import lightning.pytorch as pl
from lightning.pytorch.loggers import WandbLogger, TensorBoardLogger
from lightning.pytorch.callbacks import ModelCheckpoint


class AdaIRModel(pl.LightningModule):
    def __init__(self, args):
        super().__init__()
        self.args = args
        self.net = AdaIR(decoder=True)
        self.loss_fn = nn.L1Loss()
        self.mse_loss = nn.MSELoss()

    def forward(self, x):
        return self.net(x)

    def training_step(self, batch, batch_idx):
        # training_step defines the train loop.
        ([clean_name, de_id], degrad_patch, clean_patch) = batch
        restored = self.net(degrad_patch)

        loss = self.loss_fn(restored, clean_patch)
        self.log("train_loss", loss, prog_bar=True, logger=True)

        if self.trainer.is_last_batch:
            mse = self.mse_loss(restored, clean_patch)
            psnr = 10 * torch.log10(1.0 / mse)
            self.log("train_psnr", psnr, prog_bar=True, logger=True)

        return loss

    def validation_step(self, batch, batch_idx):
        # validation_step defines the validation loop.
        ([clean_name, de_id], degrad_patch, clean_patch) = batch
        restored = self.net(degrad_patch)

        val_loss = self.loss_fn(restored, clean_patch)
        self.log("val_loss", val_loss, prog_bar=True, logger=True)

        mse = self.mse_loss(restored, clean_patch)
        psnr = 10 * torch.log10(1.0 / mse)
        self.log("val_psnr", psnr, prog_bar=True, logger=True)

        return val_loss

    def lr_scheduler_step(self, scheduler, metric):
        scheduler.step(self.current_epoch)
        # lr = scheduler.get_lr()

    def configure_optimizers(self):
        optimizer = optim.AdamW(self.parameters(), lr=self.args.lr)
        scheduler = LinearWarmupCosineAnnealingLR(
            optimizer=optimizer, warmup_epochs=15, max_epochs=self.args.epochs)

        return [optimizer], [scheduler]


def main():
    print("Options")
    print(opt)
    if opt.wblogger is not None:
        logger = WandbLogger(project=opt.wblogger, name="AdaIR")
    else:
        logger = TensorBoardLogger(save_dir="logs/")

    trainset = TrainDataset(opt)
    checkpoint_callback_train = ModelCheckpoint(
        dirpath=opt.ckpt_dir,
        filename="train-psnr-{epoch:03d}-{train_psnr:.2f}",
        monitor="train_psnr",
        mode="max",
        save_top_k=3,
        save_last=False
    )
    checkpoint_callback_val = ModelCheckpoint(
        dirpath=opt.ckpt_dir,
        filename="val-psnr-{epoch:03d}-{val_psnr:.2f}",
        monitor="val_psnr",
        mode="max",
        save_top_k=3,
        save_last=False
    )
    trainloader = DataLoader(
        trainset,
        batch_size=opt.batch_size,
        pin_memory=True,
        shuffle=True,
        drop_last=True,
        num_workers=opt.num_workers
    )

    model = AdaIRModel(opt)

    trainer = pl.Trainer(
        max_epochs=opt.epochs,
        accelerator="gpu",
        devices=opt.num_gpus,
        # strategy="ddp_find_unused_parameters_true",
        logger=logger,
        callbacks=[checkpoint_callback_train],
        # precision='16-mixed',
        val_check_interval=1.0,
        accumulate_grad_batches=1,
    )
    trainer.fit(model=model, train_dataloaders=trainloader)


if __name__ == '__main__':
    main()
