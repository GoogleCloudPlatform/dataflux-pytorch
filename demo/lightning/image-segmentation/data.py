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

import lightning.pytorch as pl
import random
import numpy as np
from torch.utils.data import DataLoader
from torch.utils.data import RandomSampler
from torchvision import transforms

from dataset import DatafluxPytTrain


class Unet3DDataModule(pl.LightningDataModule):

    def __init__(self, args):
        super().__init__()
        self.args = args

    def prepare_data(self):
        pass

    def setup(self, stage="fit"):
        if stage == "fit":
            train_data_kwargs = {
                "patch_size": self.args.input_shape,
                "oversampling": self.args.oversampling,
                "seed": self.args.seed,
                "images_prefix": self.args.images_prefix,
                "labels_prefix": self.args.labels_prefix,
                "transforms": get_train_transforms(),
            }
            self.train_dataset = DatafluxPytTrain(
                project_name=self.args.gcp_project,
                bucket_name=self.args.gcs_bucket,
                **train_data_kwargs,
            )
            self.train_sampler = None
            if self.args.num_workers > 1:
                self.train_sampler = RandomSampler(self.train_dataset)

    def train_dataloader(self):
        return DataLoader(
            self.train_dataset,
            batch_size=self.args.batch_size,
            shuffle=not self.args.benchmark and self.train_sampler is None,
            sampler=self.train_sampler,
            num_workers=self.args.num_dataloader_threads,
            persistent_workers=True,
            pin_memory=True,
            drop_last=True,
        )


def get_train_transforms():
    rand_flip = RandFlip()
    cast = Cast(types=(np.float32, np.uint8))
    rand_scale = RandomBrightnessAugmentation(factor=0.3, prob=0.1)
    rand_noise = GaussianNoise(mean=0.0, std=0.1, prob=0.1)
    train_transforms = transforms.Compose(
        [rand_flip, cast, rand_scale, rand_noise])
    return train_transforms


class RandFlip:

    def __init__(self):
        self.axis = [1, 2, 3]
        self.prob = 1 / len(self.axis)

    def flip(self, data, axis):
        data["image"] = np.flip(data["image"], axis=axis).copy()
        data["label"] = np.flip(data["label"], axis=axis).copy()
        return data

    def __call__(self, data):
        for axis in self.axis:
            if random.random() < self.prob:
                data = self.flip(data, axis)
        return data


class Cast:

    def __init__(self, types):
        self.types = types

    def __call__(self, data):
        data["image"] = data["image"].astype(self.types[0])
        data["label"] = data["label"].astype(self.types[1])
        return data


class RandomBrightnessAugmentation:

    def __init__(self, factor, prob):
        self.prob = prob
        self.factor = factor

    def __call__(self, data):
        image = data["image"]
        if random.random() < self.prob:
            factor = np.random.uniform(low=1.0 - self.factor,
                                       high=1.0 + self.factor,
                                       size=1)
            image = (image * (1 + factor)).astype(image.dtype)
            data.update({"image": image})
        return data


class GaussianNoise:

    def __init__(self, mean, std, prob):
        self.mean = mean
        self.std = std
        self.prob = prob

    def __call__(self, data):
        image = data["image"]
        if random.random() < self.prob:
            scale = np.random.uniform(low=0.0, high=self.std)
            noise = np.random.normal(loc=self.mean,
                                     scale=scale,
                                     size=image.shape).astype(image.dtype)
            data.update({"image": image + noise})
        return data
