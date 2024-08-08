import lightning as pl

from arguments import PARSER
import time

from typing import Any, Dict, Optional

import torch
import functools
from dataflux_core import user_agent
from google.cloud import storage
from lightning.pytorch.plugins.io import CheckpointIO

import ray.train.lightning
from ray.train.torch import TorchTrainer
from ray.train.lightning import RayFSDPStrategy
from torch.distributed.fsdp.wrap import transformer_auto_wrap_policy
from torch.distributed.fsdp import ShardingStrategy, BackwardPrefetch
from transformers.models.gpt_neox.modeling_gpt_neox import GPTNeoXLayer
from ray.train import RunConfig, CheckpointConfig
from ray.train.lightning import RayLightningEnvironment, RayTrainReportCallback, prepare_trainer
import os
import time
from typing import Tuple

import torch
from lightning import Trainer
from lightning.pytorch.callbacks import ModelCheckpoint
from torch import Tensor
import pandas as pd
from datasets import load_dataset
from transformers import AutoTokenizer, AutoModelForCausalLM

class DollyV2Model(pl.LightningModule):
    def __init__(self, lr=2e-5, eps=1e-8):
        super().__init__()
        self.save_hyperparameters()
        self.lr = lr
        self.eps = eps
        self.model = AutoModelForCausalLM.from_pretrained("prajjwal1/bert-tiny", token="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")

    def forward(self, batch):
        outputs = self.model(
            batch["input_ids"],
            attention_mask=batch["attention_mask"],
            labels=batch["labels"]
        )
        return outputs.loss

    def training_step(self, batch, batch_idx):
        loss = self.forward(batch)
        self.log("train_loss", loss, prog_bar=True, on_step=True)
        return loss

    def configure_optimizers(self):
        if self.global_rank == 0:
            print(self.trainer.model)
        return torch.optim.AdamW(self.trainer.model.parameters(), lr=self.lr, eps=self.eps)

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
        if path.startswith("gs:/"):
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
        return path

    def save_checkpoint(
        self,
        checkpoint: Dict[str, Any],
        path: str,
        storage_options: Optional[Any] = None,
    ) -> None:
        print(path)
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

def split_text(batch: pd.DataFrame) -> pd.DataFrame:
    text = list(batch["text"])
    flat_text = "".join(text)
    split_text = [
        x.strip()
        for x in flat_text.split("\n")
        if x.strip() and not x.strip()[-1] == ":"
    ]
    return pd.DataFrame(split_text, columns=["text"])


def tokenize(batch: pd.DataFrame) -> dict:
    tokenizer = AutoTokenizer.from_pretrained("databricks/dolly-v2-7b", padding_side="left")
    tokenizer.pad_token = tokenizer.eos_token
    ret = tokenizer(
        list(batch["text"]),
        truncation=True,
        max_length=256,
        padding="max_length",
        return_tensors="np",
    )
    ret["labels"] = ret["input_ids"].copy()
    return dict(ret)

def train_func(config):
    lr = 2e-5
    eps = 1e-8
    batch_size_per_worker = 2
    model = DollyV2Model(lr=lr, eps=eps)
    strategy = config["strategy"]
    ckpt = DatafluxLightningCheckpoint(project_name="amundson-gke-aiml-demo",
                                           bucket_name="dataflux-checkpointing")

    train_ds = ray.train.get_dataset_shard("train")
    train_dataloader = train_ds.iter_torch_batches(batch_size=batch_size_per_worker)

    trainer = Trainer(
        default_root_dir="gs://dataflux-checkpointing/bert-tiny",
        max_epochs=config["flags"].epochs,
        devices="auto",
        max_steps=1,
        accelerator="auto",
        strategy=strategy,
        plugins=[ray.train.lightning.RayLightningEnvironment(),ckpt],
        callbacks=[RayTrainReportCallback()],
        enable_checkpointing=True,
    )
    trainer = prepare_trainer(trainer)
    trainer.fit(model, train_dataloaders=train_dataloader)

    start = time.time()
    for i in range(2):
        trainer.save_checkpoint(os.path.join("gs://dataflux-checkpointing/bert-tiny/",f'ckpt_{i}.ckpt'))
    end = time.time()
    print("Average time to save one checkpoint: " + str((end-start)/2) + " seconds")


if __name__ == "__main__":
    hf_dataset = load_dataset("tiny_shakespeare")
    train_ds = ray.data.from_huggingface(hf_dataset["train"])
    train_ds = train_ds.map_batches(split_text, batch_format="pandas")
    train_ds.take(10)
    train_ds = train_ds.map_batches(tokenize, batch_format="pandas")
    run_config = RunConfig(
        name="test-run1",
        storage_path="gs://dataflux-checkpointing",
        checkpoint_config=CheckpointConfig(),
    )
    auto_wrap_policy = functools.partial(
        transformer_auto_wrap_policy,
        transformer_layer_cls = {GPTNeoXLayer}
    )
    fsdp_strategy = RayFSDPStrategy(
        sharding_strategy=ShardingStrategy.FULL_SHARD,
        backward_prefetch=BackwardPrefetch.BACKWARD_PRE,
        forward_prefetch=True,
        auto_wrap_policy=auto_wrap_policy,
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
        datasets={"train": train_ds},
    )
    trainer.fit()
