import lightning as pl

from model import Unet3DLightning
from data import Unet3DDataModule
from arguments import PARSER

import ray.train.lightning
from ray.train.torch import TorchTrainer


def train_func(config):
    model = Unet3DLightning(config["flags"])
    train_data_loader = Unet3DDataModule(config["flags"])
    trainer = pl.Trainer(
        max_epochs=config["flags"].epochs,
        devices="auto",
        accelerator=config["flags"].accelerator,
        strategy=ray.train.lightning.RayDDPStrategy(),
        plugins=[ray.train.lightning.RayLightningEnvironment()],
        enable_checkpointing=False,
    )
    trainer = ray.train.lightning.prepare_trainer(trainer)
    trainer.fit(model=model, train_dataloaders=train_data_loader)



if __name__ == "__main__":
    flags = PARSER.parse_args()
    config = dict()
    config["flags"] = flags

    # Configure scaling and resource requirements.
    scaling_config = ray.train.ScalingConfig(num_workers=2, use_gpu=True)

    # Launch distributed training job.
    trainer = TorchTrainer(
        train_func,
        scaling_config=scaling_config,
        train_loop_config=config,
    )
    trainer.fit()

