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

import os
from math import ceil
from typing import Dict

import numpy as np
import ray
import ray.train
import torch
from arguments import PARSER
from data_loader import get_data_loaders
from filelock import FileLock
from shared.losses import DiceCELoss, DiceScore
from model.unet3d import Unet3D
from ray.air import ScalingConfig, session
from ray.train.torch import TorchTrainer, get_device
from torch import nn
from torch.cuda.amp import GradScaler, autocast
from torch.optim import SGD, Adam
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from torchvision.transforms import Normalize, ToTensor
from tqdm import tqdm


def get_optimizer(params, flags):
    if flags.optimizer == "adam":
        optim = Adam(params,
                     lr=flags.learning_rate,
                     weight_decay=flags.weight_decay)
    elif flags.optimizer == "sgd":
        optim = SGD(
            params,
            lr=flags.learning_rate,
            momentum=flags.momentum,
            nesterov=True,
            weight_decay=flags.weight_decay,
        )
    elif flags.optimizer == "lamb":
        import apex

        optim = apex.optimizers.FusedLAMB(
            params,
            lr=flags.learning_rate,
            betas=flags.lamb_betas,
            weight_decay=flags.weight_decay,
        )
    else:
        raise ValueError("Optimizer {} unknown.".format(flags.optimizer))
    return optim


def lr_warmup(optimizer, init_lr, lr, current_epoch, warmup_epochs):
    scale = current_epoch / warmup_epochs
    for param_group in optimizer.param_groups:
        param_group["lr"] = init_lr + (lr - init_lr) * scale


def train_func_per_worker(config: Dict):
    lr = config["lr"]
    epochs = config["epochs"]
    batch_size = config["batch_size_per_worker"]
    flags = config["flags"]
    num_workers = config["num_workers"]

    # Get dataloaders inside worker training function.
    train_dataloader = get_data_loaders(flags,
                                        num_shards=num_workers,
                                        global_rank=session.get_world_rank())

    # Each worker is assigned one GPU so get_device will return one device.
    device = get_device()

    model = Unet3D(1,
                   3,
                   normalization=flags.normalization,
                   activation=flags.activation)

    # Prepare model for distributed training.
    model = ray.train.torch.prepare_model(model)

    optimizer = get_optimizer(model.parameters(), flags)
    if flags.lr_decay_epochs:
        scheduler = torch.optim.lr_scheduler.MultiStepLR(
            optimizer,
            milestones=flags.lr_decay_epochs,
            gamma=flags.lr_decay_factor)
    scaler = GradScaler()
    loss_fn = DiceCELoss(
        to_onehot_y=True,
        use_softmax=True,
        layout=flags.layout,
        include_background=flags.include_background,
    )
    score_fn = DiceScore(
        to_onehot_y=True,
        use_argmax=True,
        layout=flags.layout,
        include_background=flags.include_background,
    )

    # Model training loop.
    for epoch in range(epochs):
        model.train()
        if epoch <= flags.lr_warmup_epochs and flags.lr_warmup_epochs > 0:
            lr_warmup(
                optimizer,
                flags.init_learning_rate,
                flags.learning_rate,
                epoch,
                flags.lr_warmup_epochs,
            )
        train_dataloader.sampler.set_epoch(epoch)

        for image, label in tqdm(train_dataloader,
                                 desc=f"Train Epoch {epoch}"):
            image = image.to(device)
            label = label.to(device)
            pred = model(image)  # .to(device)
            loss = loss_fn(pred, label)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        ray.air.session.report(metrics={"train_loss": loss.item()})


def train_unet(flags, num_workers=2, use_gpu=True):
    train_config = {
        "lr": flags.learning_rate,
        "epochs": flags.epochs,
        "batch_size_per_worker": flags.batch_size,
        "flags": flags,
        "num_workers": num_workers,
    }

    # Configure computation resources.
    scaling_config = ScalingConfig(
        num_workers=num_workers,
        use_gpu=True,
        # We have 32 CPUs and 4 GPUs per worker VM, so we can create a total of 8 individual workers each with 1 GPU for training.
        resources_per_worker={
            "CPU": 8,
            "GPU": 1
        },
    )

    # Initialize a Ray TorchTrainer.
    trainer = TorchTrainer(
        train_loop_per_worker=train_func_per_worker,
        train_loop_config=train_config,
        scaling_config=scaling_config,
    )

    # Start distributed training: run `train_func_per_worker` on all workers.
    result = trainer.fit()
    print(f"Training result: {result}")


if __name__ == "__main__":
    flags = PARSER.parse_args()
    train_unet(flags, num_workers=flags.num_workers, use_gpu=True)
