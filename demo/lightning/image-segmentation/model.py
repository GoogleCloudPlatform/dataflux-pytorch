import torch
import lightning.pytorch as pl


class Unet3D(pl.LightningModule):
    def __init__(self):
        pass
    
    def forward(self, x):
        pass
    
    def configure_optimizers(self):
        pass

    def training_step(self, train_batch, batch_idx):
        pass

    def backward(self, trainer, loss, optimizer, optimizer_idx):
        pass

    def optimizer_step(self, epoch, batch_idx, optimizer, optimizer_idx):
        pass
