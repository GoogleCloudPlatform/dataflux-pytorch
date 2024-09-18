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
import socket
import time

from lightning import Trainer
from lightning.pytorch.callbacks import ModelCheckpoint
from lightning.pytorch.demos import (LightningTransformer, Transformer,
                                     WikiText2)
from torch.utils.data import DataLoader

from dataflux_pytorch.lightning import DatafluxLightningCheckpoint


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--project", type=str)
    parser.add_argument("--ckpt-path", type=str)
    parser.add_argument("--prefix", type=str)
    parser.add_argument("--save-only-latest",
                        action="store_true",
                        default=False)
    parser.add_argument("--min-epochs", type=int, default=2)
    parser.add_argument("--max-epochs", type=int, default=5)
    parser.add_argument("--max-steps", type=int, default=3)
    parser.add_argument("--num-workers", type=int, default=1)
    parser.add_argument("--num-layers", type=int, default=1000)
    return parser.parse_args()


def main():
    args = parse_args()
    dataset = WikiText2()
    dataloader = DataLoader(dataset, num_workers=args.num_workers)

    model = DemoTransformer(vocab_size=dataset.vocab_size,
                            nlayers=args.num_layers)
    dataflux_ckpt = DatafluxLightningCheckpoint(project_name=args.project)
    # Save once per step, and if `save_only_latest`, replace the last checkpoint each time.
    # Replacing is implemented by saving the new checkpoint, and then deleting the previous one.
    # If `save_only_latest` is False, a new checkpoint is created for each step.
    checkpoint_callback = ModelCheckpoint(
        save_top_k=1 if args.save_only_latest else -1,
        every_n_train_steps=1,
        filename="checkpoint-{epoch:02d}-{step:02d}",
        enable_version_counter=True,
    )
    trainer = Trainer(default_root_dir=args.ckpt_path,
                      plugins=[dataflux_ckpt],
                      callbacks=[checkpoint_callback],
                      min_epochs=args.min_epochs,
                      max_epochs=args.max_epochs,
                      max_steps=args.max_steps,
                      accelerator="cpu",
                      strategy="ddp",
                      num_nodes=1)
    t1 = time.time()
    trainer.fit(model, dataloader)
    t2 = time.time()
    result_time = t2 - t1
    print(f"Total elapsed time was {result_time} seconds...")


class DemoTransformer(LightningTransformer):

    def __init__(
        self,
        vocab_size: int = 33278,
        nlayers: int = 2,
    ) -> None:
        super().__init__()
        self.model = Transformer(vocab_size=vocab_size, nlayers=nlayers)


if __name__ == "__main__":
    main()
