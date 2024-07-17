import lightning as pl

from model import Unet3DLightning
from data import Unet3DDataModule

if __name__ == "__main__":
    # hard coded args for testing.
    args = {
        "a": 1,
        "b": 2,
    }
    model = Unet3DLightning(args)
    train_data_loader = Unet3DDataModule()
    trainer = pl.Trainer(devices=2, accelerator="gpu", max_epochs=5)
    trainer.fit(model=model, train_dataloaders=train_data_loader)
