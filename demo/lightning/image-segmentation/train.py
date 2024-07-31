import lightning as pl

from model import Unet3DLightning
from data import Unet3DDataModule
from arguments import PARSER
import time

from typing import Any, Dict, Optional

import torch
from dataflux_core import user_agent
from google.cloud import storage
from lightning.pytorch.plugins.io import CheckpointIO

import ray.train.lightning
from ray.train.torch import TorchTrainer
from ray.train.lightning import RayFSDPStrategy
from torch.distributed.fsdp.wrap import transformer_auto_wrap_policy
from torch.distributed.fsdp import ShardingStrategy, BackwardPrefetch
from transformers.models.gpt_neox.modeling_gpt_neox import GPTNeoXLayer
import dataflux_pytorch
from ray.train import RunConfig, CheckpointConfig

class DatafluxLightningCheckpoint(CheckpointIO):
    """A checkpoint manager for GCS using the :class:'CheckpointIO' interface"""

    def __init__(
        self,
        project_name: str,
        bucket_name: str,
        storage_client: Optional[storage.Client] = None,
    ):
        self.project_name = project_name
        self.bucket_name = bucket_name
        self.storage_client = storage_client
        if not storage_client:
            self.storage_client = storage.Client(project=self.project_name, )
        user_agent.add_dataflux_user_agent(self.storage_client)
        self.bucket = self.storage_client.bucket(self.bucket_name)

    def _parse_gcs_path(self, path: str) -> str:
        if not path or not (path.startswith("gcs://")
                            or path.startswith("gs://")):
            raise ValueError("Path needs to begin with gcs:// or gs://")
        path = path.split("//", maxsplit=1)
        if not path or len(path) < 2:
            raise ValueError("Bucket name must be non-empty")
        split = path[1].split("/", maxsplit=1)
        if len(split) == 1:
            bucket = split[0]
            prefix = ""
        else:
            bucket, prefix = split
        if not bucket:
            raise ValueError("Bucket name must be non-empty")
        if bucket != self.bucket_name:
            raise ValueError(
                f'Unexpected bucket name, expected {self.bucket_name} got {bucket}'
            )
        return prefix

    def save_checkpoint(
        self,
        checkpoint: Dict[str, Any],
        path: str,
        storage_options: Optional[Any] = None,
    ) -> None:
        key = self._parse_gcs_path(path)
        blob = self.bucket.blob(key)
        with blob.open("wb", ignore_flush=True) as blobwriter:
            torch.save(checkpoint, blobwriter)

    def load_checkpoint(
        self,
        path: str,
        map_location: Optional[Any] = None,
    ) -> Dict[str, Any]:
        key = self._parse_gcs_path(path)
        blob = self.bucket.blob(key)
        return torch.load(blob.open("rb"), map_location)

    def remove_checkpoint(
        self,
        path: str,
    ) -> None:
        key = self._parse_gcs_path(path)
        blob = self.bucket.blob(key)
        blob.delete()

    def teardown(self, ) -> None:
        pass


def train_func(config):
    model = Unet3DLightning(config["flags"])
    train_data_loader = Unet3DDataModule(config["flags"])
    ckpt = DatafluxLightningCheckpoint(project_name="amundson-gke-aiml-demo",bucket_name="dataflux-checkpointing")
    strategy = config["strategy"]
    trainer = pl.Trainer(
        default_root_dir="gs://dataflux-checkpointing/trail1",
        max_epochs=config["flags"].epochs,
        devices="auto",
        max_steps=1,
        accelerator="auto",
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
    scaling_config = ray.train.ScalingConfig(num_workers=1, use_gpu=True)
    # Launch distributed training job.
    trainer = TorchTrainer(
        train_func,
        scaling_config=scaling_config,
        run_config=run_config,
        train_loop_config=config,
    )
    trainer.fit()

