import lightning as pl

from model import Unet3DLightning
from data import Unet3DDataModule
from arguments import PARSER
import time

import ray.train.lightning
from ray.train.torch import TorchTrainer
from ray.train.lightning import RayFSDPStrategy
from torch.distributed.fsdp.wrap import transformer_auto_wrap_policy
from torch.distributed.fsdp import ShardingStrategy, BackwardPrefetch
from transformers.models.gpt_neox.modeling_gpt_neox import GPTNeoXLayer
import dataflux_pytorch
from ray.train import RunConfig, CheckpointConfig


def train_func(config):
    model = Unet3DLightning(config["flags"])
    train_data_loader = Unet3DDataModule(config["flags"])
    ckpt = dataflux_pytorch.lightning.DatafluxLightningCheckpoint(project_name="amundson-gke-aiml-demo",bucket_name="dataflux-checkpointing")
    strategy = config["strategy"]

    trainer = pl.Trainer(
        default_root_dir="gs://dataflux-checkpointing/trail1",
        max_epochs=config["flags"].epochs,
        devices="auto",
        max_steps=1,
        accelerator=config["flags"].accelerator,
        strategy=strategy,
        plugins=[ray.train.lightning.RayLightningEnvironment(),ckpt],
        enable_checkpointing=True,
    )
    trainer = ray.train.lightning.prepare_trainer(trainer)
    trainer.fit(model=model, train_dataloaders=train_data_loader)
    start = time.time()
    trainer.save_checkpoint("gs://dataflux-checkpointing/trail1/ckpt.ckpt")
    end = time.time()
    print(end-start)

if __name__ == "__main__":
    run_config = RunConfig(
        name="test-run1",
        storage_path="gs://dataflux-checkpointing",
        checkpoint_config=CheckpointConfig(),
    )

    fsdp_strategy = RayFSDPStrategy(
        sharding_strategy=ShardingStrategy.FULL_SHARD,
        backward_prefetch=BackwardPrefetch.BACKWARD_PRE,
        forward_prefetch=True,
        limit_all_gathers=True,
        activation_checkpointing=[GPTNeoXLayer],
    )
    flags = PARSER.parse_args()
    config = dict()
    config["flags"] = flags
    config["strategy"] = fsdp_strategy

    # Configure scaling and resource requirements.
    scaling_config = ray.train.ScalingConfig(num_workers=2, use_gpu=True)

    # Launch distributed training job.
    trainer = TorchTrainer(
        train_func,
        scaling_config=scaling_config,
        run_config=run_config,
        train_loop_config=config,
    )
    trainer.fit()

