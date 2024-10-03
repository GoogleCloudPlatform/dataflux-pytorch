"""
 Copyright 2024 Google LLC

 Licensed under the Apache License, Version 2.0 (the "License");
 you may not use this file except in compliance with the License.
 You may obtain a copy of the License at

      https://www.apache.org/licenses/LICENSE-2.0

 Unless required by applicable law or agreed to in writing, software
 distributed under the License is distributed on an "AS IS" BASIS,
 WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 See the License for the specific language governing permissions and
 limitations under the License.
 """
import argparse
import os
import sys
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

from dataflux_pytorch.lightning import (DatafluxLightningAsyncCheckpoint,
                                        DatafluxLightningCheckpoint)


class BenchmarkDatafluxLightningAsyncCheckpoint(
        DatafluxLightningAsyncCheckpoint):

    def teardown(self, *args, **kwargs):
        # Prevent parent teardown from terminating the executor after fit.
        pass

    def finalize(self, *args, **kwargs):
        # Provide a different invocation of the teardown process.
        super().teardown()


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


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--project", type=str)
    parser.add_argument("--ckpt-dir-path", type=str)
    parser.add_argument("--save-only-latest",
                        action="store_true",
                        default=False)
    parser.add_argument("--layers", type=int, default=100)
    parser.add_argument("--steps", type=int, default=5)
    parser.add_argument("--disable-multipart",
                        action="store_true",
                        default=False)
    parser.add_argument("--clear-kernel-cache",
                        action="store_true",
                        default=False)
    parser.add_argument(
        '--checkpoint',
        choices=['checkpointio', 'asynccheckpointio', 'no-dataflux'],
        default='checkpointio')
    return parser.parse_args()


def main():
    """Checkpoints a PyTorch Ligthning demo model to GCS using gcsfs or DatafluxLightningCheckpoint.

    This function utilizes PyTorch Lightning to checkpoint the WikiText2 dataset. It
    takes in information regarding the gcs location to save the checkpoints, the type of
    checkpoint, and other configuration variables. Default this function runs on
    gcsfs to write PyTorch Ligthtning checkpoints, TorchCheckpointIO. If dataflux_ckpt
    is enabled the Trainer will be passed a DatafluxLightningCheckpoint, which is an
    implementation of the CheckpointIO interface, as a plugin.

    Typical usage example:

      Run DatafluxLightningCheckpoint over 10 steps:

      python3 train.py --project=my-project --ckpt_dir_path=gs://bucket-name/path/to/dir/ --save_only_latest --layers=1000 --steps=10

      Run gcsfs over 10 steps:

      python3 train.py --project=my-project --ckpt_dir_path=gs://bucket-name/path/to/dir/ --layers=1000 --steps=10 --checkpoint=no-dataflux

    """
    args = parse_args()
    if args.steps < 1:
        raise ValueError("Steps need to greater than 0.")

    dataset = WikiText2()
    dataloader = DataLoader(dataset, num_workers=1)
    model = LightningTransformer(vocab_size=dataset.vocab_size,
                                 nlayers=args.layers)

    # Checkpoint strategy selection.
    if args.checkpoint == 'checkpointio':
        ckpt = DatafluxLightningCheckpoint(project_name=args.project)
    elif args.checkpoint == 'asynccheckpointio':
        print("NOTE: AsyncCheckpointIO is enabled.")
        ckpt = BenchmarkDatafluxLightningAsyncCheckpoint(
            project_name=args.project)
    elif args.checkpoint == 'no-dataflux':
        ckpt = TorchCheckpointIO()
    else:
        raise ValueError("Invalid choice for --checkpoint")

    # Save once per step, and if `save_only_latest`, replace the last checkpoint each time.
    # Replacing is implemented by saving the new checkpoint, and then deleting the previous one.
    # If `save_only_latest` is False, a new checkpoint is created for each step.
    checkpoint_callback = ModelCheckpoint(
        save_top_k=1 if args.save_only_latest else -1,
        every_n_train_steps=1,
        filename="checkpoint-{epoch:02d}-{step:02d}",
        enable_version_counter=True,
    )
    trainer = Trainer(
        default_root_dir=args.ckpt_dir_path,
        plugins=[ckpt],
        callbacks=[checkpoint_callback],
        min_epochs=4,
        max_epochs=5,
        max_steps=1,
        accelerator="cpu",
    )
    trainer.fit(model, dataloader)

    # Measure save checkpoint.
    start = time.time()
    for i in range(args.steps):
        trainer.save_checkpoint(
            os.path.join(args.ckpt_dir_path, f'ckpt_{i}.ckpt'))
    end = time.time()
    # command to clear kernel cache only works on MacOs and Linux.
    if args.clear_kernel_cache and sys.platform in [
            "darwin", "linux", "linux2", "linux3"
    ]:
        print("Clearing kernel cache...")
        os.system("sync && sudo sysctl -w vm.drop_caches=3")
    print("Average time to save one checkpoint: " +
          str((end - start) / args.steps) + " seconds")

    # If using async, call finalize to shut down the threadpool executor.
    if args.checkpoint == 'asynccheckpointio':
        ckpt.finalize()

    # Measure load checkpoint.
    start = time.time()
    for i in range(args.steps):
        _ = ckpt.load_checkpoint(
            os.path.join(args.ckpt_dir_path, f'ckpt_{i}.ckpt'))
    end = time.time()
    print("Average time to load one checkpoint: " +
          str((end - start) / args.steps) + " seconds")


if __name__ == "__main__":
    main()
