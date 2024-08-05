import lightning as pl

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
from torch.distributed.fsdp import ShardingStrategy, BackwardPrefetch
from transformers.models.gpt_neox.modeling_gpt_neox import GPTNeoXLayer
from ray.train import RunConfig, CheckpointConfig
import os
import time
from typing import Tuple

import torch
from lightning import Trainer
from lightning.pytorch import LightningModule
from lightning.pytorch.callbacks import ModelCheckpoint
from lightning.pytorch.demos import Transformer, WikiText2
from lightning.pytorch.plugins.io import TorchCheckpointIO
from torch import Tensor
from torch.utils.data import DataLoader

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
        path = path.replace("gs:/","gs://")
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
        key = self._parse_gcs_path(str(path))
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

class LightningTransformer(LightningModule):

    def __init__(self, vocab_size: int = 33278, nlayers: int = 100) -> None:
        super().__init__()
        self.model = Transformer(vocab_size=vocab_size, nlayers=nlayers)

    def forward(self, inputs: Tensor, target: Tensor) -> Tensor:
        return self.model(inputs, target)

    def training_step(self, batch: Tuple[Tensor, Tensor],
                      batch_idx: int) -> Tensor:
        inputs, target = batch
        output = self(inputs, target)
        loss = torch.nn.functional.nll_loss(output, target.view(-1))
        return loss

    def configure_optimizers(self) -> torch.optim.Optimizer:
        return torch.optim.SGD(self.model.parameters(), lr=0.1)

    def prepare_data(self) -> None:
        WikiText2(download=True)

    def train_dataloader(self) -> DataLoader:
        dataset = WikiText2()
        return DataLoader(dataset)

def train_func(config):
    dataset = WikiText2()
    dataloader = DataLoader(dataset, num_workers=1)
    model = LightningTransformer(vocab_size=dataset.vocab_size, nlayers=100)
    strategy = config["strategy"]
    ckpt = DatafluxLightningCheckpoint(project_name="amundson-gke-aiml-demo",
                                           bucket_name="dataflux-checkpointing")
    checkpoint_callback = ModelCheckpoint(
        save_top_k=-1,
        every_n_train_steps=1,
        filename="checkpoint-{epoch:02d}-{step:02d}",
        enable_version_counter=True,
    )
    trainer = Trainer(
        default_root_dir="gs://dataflux-checkpointing/fsdp/single-node-test",
        max_epochs=config["flags"].epochs,
        devices="auto",
        max_steps=1,
        accelerator="auto",
        strategy=strategy,
        plugins=[ray.train.lightning.RayLightningEnvironment(),ckpt],
        enable_checkpointing=True,
        callbacks=[checkpoint_callback],

    )
    trainer.fit(model, dataloader)

    start = time.time()
    for i in range(2):
        trainer.save_checkpoint(os.path.join("gs://dataflux-checkpointing/fsdp/single-node-testt",f'ckpt_{i}.ckpt'))
    end = time.time()
    print("Average time to save one checkpoint: " + str((end-start)/2) + " seconds")


if __name__ == "__main__":
    run_config = RunConfig(
        name="fsdp-single-node",
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

