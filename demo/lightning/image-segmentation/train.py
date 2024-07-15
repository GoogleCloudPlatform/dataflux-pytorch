import lightning as pl

from model import Unet3D
from data import Unet3DDataModule

if __name__ == "__main__":
    model = Unet3D()
    train_data_loader = Unet3DDataModule()
    trainer = pl.Trainer(
        devices=2, 
        accelerator="gpu", 
        max_epochs=5
    )
    trainer.fit(model=model, train_dataloaders=train_data_loader)
