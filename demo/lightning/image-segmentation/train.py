import lightning as pl

from model import Unet3DLightning
from data import Unet3DDataModule
from arguments import PARSER

if __name__ == "__main__":
    flags = PARSER.parse_args()
    model = Unet3DLightning(flags)
    train_data_loader = Unet3DDataModule()
    trainer = pl.Trainer(devices=2, accelerator="gpu", max_epochs=5)
    trainer.fit(model=model, train_dataloaders=train_data_loader)
