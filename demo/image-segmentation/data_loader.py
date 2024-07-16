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
import glob

import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from torch.utils.data.dataloader import default_collate
from torch.utils.data.distributed import DistributedSampler

from dataflux_pytorch import dataflux_mapstyle_dataset
from pytorch_loader import DatafluxPytTrain


def list_files_with_pattern(path, files_pattern):
    data = sorted(glob.glob(os.path.join(path, files_pattern)))
    assert len(data) > 0, f"Found no data at {path}"
    return data


def load_data(path, files_pattern):
    data = sorted(glob.glob(os.path.join(path, files_pattern)))
    assert len(data) > 0, f"Found no data at {path}"
    return data


def get_split(data, train_idx, val_idx):
    train = list(np.array(data)[train_idx])
    val = list(np.array(data)[val_idx])
    return train, val


def split_eval_data(x_val, y_val, num_shards, shard_id):
    x = [a.tolist() for a in np.array_split(x_val, num_shards)]
    y = [a.tolist() for a in np.array_split(y_val, num_shards)]
    return x[shard_id], y[shard_id]


def get_data_split(path: str, num_shards: int, shard_id: int):
    imgs = load_data(os.path.join(path, "images"), "*_x.npy")
    lbls = load_data(os.path.join(path, "labels"), "*_y.npy")
    assert len(imgs) == len(
        lbls
    ), f"Found {len(imgs)} volumes but {len(lbls)} corresponding masks"
    images, labels = [], []
    for case_img, case_lbl in zip(imgs, lbls):
        images.append(case_img)
        labels.append(case_lbl)
    return images, labels


class SyntheticDataset(Dataset):
    def __init__(
        self,
        channels_in=1,
        channels_out=3,
        shape=(128, 128, 128),
        device="cpu",
        layout="NCDHW",
        scalar=False,
    ):
        shape = tuple(shape)
        x_shape = (
            (channels_in,) + shape if layout == "NCDHW" else shape + (channels_in,)
        )
        self.x = torch.rand(
            (32, *x_shape), dtype=torch.float32, device=device, requires_grad=False
        )
        if scalar:
            self.y = torch.randint(
                low=0,
                high=channels_out - 1,
                size=(32, *shape),
                dtype=torch.int32,
                device=device,
                requires_grad=False,
            )
            self.y = torch.unsqueeze(
                self.y, dim=1 if layout == "NCDHW" else -1)
        else:
            y_shape = (
                (channels_out,) + shape
                if layout == "NCDHW"
                else shape + (channels_out,)
            )
            self.y = torch.rand(
                (32, *y_shape), dtype=torch.float32, device=device, requires_grad=False
            )

    def __len__(self):
        return 64

    def __getitem__(self, idx):
        return self.x[idx % 32], self.y[idx % 32]


def get_data_loaders(flags, num_shards, global_rank):
    if flags.loader == "synthetic":
        train_dataset = SyntheticDataset(
            scalar=True, shape=flags.input_shape, layout=flags.layout
        )
        val_dataset = SyntheticDataset(
            scalar=True, shape=flags.val_input_shape, layout=flags.layout
        )

    elif flags.loader == "pytorch":
        train_data_kwargs = {
            "patch_size": flags.input_shape,
            "oversampling": flags.oversampling,
            "seed": flags.seed,
            "images_prefix": flags.images_prefix,
            "labels_prefix": flags.labels_prefix,
        }
        train_dataset = DatafluxPytTrain(
            project_name=flags.gcp_project,
            bucket_name=flags.gcs_bucket,
            config=dataflux_mapstyle_dataset.Config(sort_listing_results=True),
            **train_data_kwargs,
        )

    else:
        raise ValueError(
            f"Loader {flags.loader} unknown. Valid loaders are: synthetic, pytorch"
        )

    # The DistributedSampler seed should be the same for all workers.
    train_sampler = (
        DistributedSampler(train_dataset, seed=flags.seed, drop_last=True)
        if num_shards > 1
        else None
    )
    val_sampler = None

    train_dataloader = DataLoader(
        train_dataset,
        collate_fn=collate_fn,
        batch_size=flags.batch_size,
        shuffle=not flags.benchmark and train_sampler is None,
        sampler=train_sampler,
        num_workers=flags.num_dataloader_threads,
        pin_memory=True,
        drop_last=True,
    )

    return train_dataloader


def collate_fn(batch):
    batch = list(
        filter(lambda x: x["image"] is not None and x["label"] is not None, batch))
    return default_collate(batch)
