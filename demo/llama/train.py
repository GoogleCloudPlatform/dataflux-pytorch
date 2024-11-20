# Copyright Lightning AI. Licensed under the Apache License 2.0, see LICENSE file.

# Borrowed from https://github.com/Lightning-AI/lit-llama/blob/main/pretrain/redpajama.py
# with changes at appropriate places to use GCS Connector for Pytorch for data listing, loading, and checkpointing.

import os
import sys
import time
from functools import partial
from pathlib import Path
from typing import Optional

import lightning as L
import torch
from google.cloud import storage
from lit_llama.model import Block, LLaMA, LLaMAConfig
from pretrain.redpajama import get_lr, validate
from torch.distributed.fsdp.wrap import transformer_auto_wrap_policy
from torch.utils.data import DataLoader

from .strategies import DatafluxFSDPStrategy

# dataflux vars
project_name = "<YOUR-PROJECT>"
bucket_name = "<YOUR-BUCKET>"
checkpoint_save_dir = "<YOUR-BUCKET>"

# support running without installing as a package
wd = Path(__file__).parent.parent.resolve()
sys.path.append(str(wd))

save_interval = 1000
eval_interval = 1000
eval_iters = 100
log_interval = 1

# compile = False

# Hyperparameters
learning_rate = 6e-4
batch_size = 125
micro_batch_size = 5
max_iters = 600000  # num_epochs * (epoch_size // micro_batch_size) // devices
weight_decay = 1e-1
beta1 = 0.9
beta2 = 0.95
grad_clip = 1.0
decay_lr = True
warmup_iters = 2000
lr_decay_iters = max_iters
min_lr = 6e-5


def main(
    devices: int = 4,
    train_data_dir: Path = "data/lit-redpajama",
    val_data_dir: Optional[Path] = None,
) -> None:
    auto_wrap_policy = partial(transformer_auto_wrap_policy,
                               transformer_layer_cls={Block})
    strategy = DatafluxFSDPStrategy(
        project_name=project_name,
        storage_client=storage.Client(project=project_name),
        auto_wrap_policy=auto_wrap_policy,
        activation_checkpointing=Block,
        limit_all_gathers=True,
    )

    fabric = L.Fabric(accelerator="gpu",
                      devices=devices,
                      precision="bf16-mixed",
                      strategy=strategy)
    fabric.launch()
    fabric.seed_everything(1337)

    config = LLaMAConfig.from_name("7B")

    train_dataloader, val_dataloader = create_dataloaders(
        batch_size=micro_batch_size,
        block_size=config.block_size,
        fabric=fabric,
        train_data_dir=train_data_dir,
        val_data_dir=val_data_dir,
        seed=1338,
    )
    if val_dataloader is None:
        train_dataloader = fabric.setup_dataloaders(train_dataloader)
    else:
        train_dataloader, val_dataloader = fabric.setup_dataloaders(
            train_dataloader, val_dataloader)

    with fabric.device:
        torch.set_default_dtype(torch.bfloat16)
        model = LLaMA(config)
        model.apply(model._init_weights)
        torch.set_default_dtype(torch.float32)

    # if compile:
    #     model = torch.compile(model)

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=learning_rate,
        weight_decay=weight_decay,
        betas=(beta1, beta2),
        foreach=False,
    )

    model, optimizer = fabric.setup(model, optimizer)

    process_batch_size = batch_size // devices
    gradient_accumulation_iters = process_batch_size // micro_batch_size

    train(fabric, model, optimizer, train_dataloader, val_dataloader,
          gradient_accumulation_iters, devices)


def train(
    fabric: L.Fabric,
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    train_dataloader: DataLoader,
    val_dataloader: Optional[DataLoader],
    grad_accum_steps: int,
    devices: int,
) -> None:
    """The training loop.

    Loosely based on the nanoGPT implementation: https://github.com/karpathy/nanoGPT.
    """

    step_count = 0

    step_time = 0.0
    tokens = 0
    prev_t1 = time.time()

    for iter_num, train_data in enumerate(train_dataloader):
        t0 = time.time()

        # determine and set the learning rate for this iteration
        lr = get_lr(iter_num) if decay_lr else learning_rate
        for param_group in optimizer.param_groups:
            param_group["lr"] = lr

        input_ids = train_data[:, 0:model.config.block_size].contiguous()
        targets = train_data[:, 1:model.config.block_size + 1].contiguous()

        is_accumulating = (iter_num + 1) % grad_accum_steps != 0

        with fabric.no_backward_sync(model, enabled=is_accumulating):
            logits = model(input_ids)
            loss = torch.nn.functional.cross_entropy(logits.view(
                -1, logits.size(-1)),
                                                     targets.view(-1),
                                                     ignore_index=-1)
            fabric.backward(loss / grad_accum_steps)

        t1 = time.time()

        if not is_accumulating:
            fabric.clip_gradients(model, optimizer, max_norm=grad_clip)

            optimizer.step()
            optimizer.zero_grad()
            step_count += 1

            t1 = time.time()

            if val_dataloader is not None and step_count % eval_interval == 0:
                val_loss = validate(fabric, model, val_dataloader)
                fabric.print(f"step {iter_num}: val loss {val_loss:.4f}")
                fabric.barrier()
                fabric.log_dict({
                    "iter": iter_num,
                    "val_loss": val_loss,
                    "step": step_count,
                    "lr": lr
                })

            if step_count % save_interval == 0:
                cur_ckpt_save_dir = os.path.join(checkpoint_save_dir,
                                                 f"iter-{iter_num:06d}")
                fabric.print(f"Saving checkpoint to {cur_ckpt_save_dir}")
                state = {
                    "model": model,
                    "optimizer": optimizer,
                    "iteration": iter_num
                }
                save_start = time.time()
                fabric.save(cur_ckpt_save_dir, state)
                save_end = time.time()

                if fabric.global_rank == 0:
                    fabric.print(
                        f"Checkpoint save with dataflux took {save_end - save_start} seconds."
                    )

        dt = t1 - t0

        tokens += micro_batch_size * model.config.block_size
        step_time += t1 - prev_t1
        prev_t1 = t1

        if iter_num % log_interval == 0:
            tokens_sec_str = f"{tokens / step_time:.0f}" if not is_accumulating else "-"

            fabric.log_dict({
                "iter": iter_num,
                "train_loss": loss,
                "step": step_count,
                "lr": lr
            })
            fabric.print(
                f"iter {iter_num}: loss {loss.item():.4f}, time: {dt*1000:.2f}ms, speed: {tokens_sec_str} toks/s/device"
            )

        if not is_accumulating:
            tokens = 0
            step_time = 0.0

        if iter_num > max_iters:
            break


if __name__ == "__main__":
    # Uncomment this line if you see an error: "Expected is_sm80 to be true, but got false"
    # torch.backends.cuda.enable_flash_sdp(False)
    torch.set_float32_matmul_precision("high")

    from jsonargparse.cli import CLI

    CLI(main)
