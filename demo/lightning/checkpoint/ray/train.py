from arguments import PARSER
import time
import functools

import ray.train.lightning
from ray.train.torch import TorchTrainer
from ray.train.lightning import RayFSDPStrategy
from transformers.models.gpt_neox.modeling_gpt_neox import GPTNeoXLayer
from ray.train import RunConfig, CheckpointConfig
from ray.train.lightning import RayLightningEnvironment, RayTrainReportCallback, prepare_trainer

import lightning as pl
from lightning import Trainer
import torch
from torch.distributed.fsdp.wrap import transformer_auto_wrap_policy
from torch.distributed.fsdp import BackwardPrefetch
import pandas as pd
from datasets import load_dataset
from transformers import AutoTokenizer, AutoModelForCausalLM

from dataflux_pytorch.lightning import DatafluxLightningCheckpoint

class BertTinyModel(pl.LightningModule):
    def __init__(self, lr=2e-5, eps=1e-8):
        super().__init__()
        self.save_hyperparameters()
        self.lr = lr
        self.eps = eps
        self.model = AutoModelForCausalLM.from_pretrained("prajjwal1/bert-tiny")

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
    batch_size_per_worker = config["flags"].batch_size
    model = BertTinyModel(lr=lr, eps=eps)
    strategy = config["strategy"]
    ckpt = DatafluxLightningCheckpoint(project_name=config["flags"].gcp_project,
                                           bucket_name=config["flags"].gcs_bucket)

    train_ds = ray.train.get_dataset_shard("train")
    train_dataloader = train_ds.iter_torch_batches(batch_size=batch_size_per_worker)

    trainer = Trainer(
        default_root_dir=config["flags"].default_root_dir,
        max_epochs=config["flags"].epochs,
        devices="auto",
        max_steps=1,
        accelerator="auto",
        strategy=strategy,
        plugins=[RayLightningEnvironment(),ckpt],
        callbacks=[RayTrainReportCallback()],
        enable_checkpointing=True,
    )
    trainer = prepare_trainer(trainer)
    trainer.fit(model, train_dataloaders=train_dataloader)

    start = time.time()
    print(config["flags"].save_ckpt_path)
    trainer.save_checkpoint("test.ckpt")
    end = time.time()
    print("Time to save distributed checkpoint: " + str(end-start) + " seconds")


if __name__ == "__main__":
    hf_dataset = load_dataset("tiny_shakespeare")
    train_ds = ray.data.from_huggingface(hf_dataset["train"])
    train_ds = train_ds.map_batches(split_text, batch_format="pandas")
    train_ds.take(10)
    train_ds = train_ds.map_batches(tokenize, batch_format="pandas")
    auto_wrap_policy = functools.partial(
        transformer_auto_wrap_policy,
        transformer_layer_cls = {GPTNeoXLayer}
    )
    fsdp_strategy = RayFSDPStrategy(
        state_dict_type="sharded",
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
    run_config = RunConfig(
        name=config["flags"].run_name,
        storage_path=config["flags"].default_root_dir,
        checkpoint_config=CheckpointConfig(),
    )
    # Configure scaling and resource requirements.
    scaling_config = ray.train.ScalingConfig(num_workers=config["flags"].num_workers, use_gpu=True)
    # Launch distributed training job.
    trainer = TorchTrainer(
        train_func,
        scaling_config=scaling_config,
        run_config=run_config,
        train_loop_config=config,
        datasets={"train": train_ds},
    )
    trainer.fit()
