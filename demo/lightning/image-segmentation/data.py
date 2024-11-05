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
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler
from demo.shared.image_segmentation import get_train_transforms

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
                self.train_sampler = DistributedSampler(self.train_dataset,
                                                        seed=self.args.seed,
                                                        drop_last=True)

    def train_dataloader(self):
        return DataLoader(
            self.train_dataset,
            batch_size=self.args.batch_size,
            shuffle=not self.args.benchmark and self.train_sampler is None,
            sampler=self.train_sampler,
            num_workers=self.args.num_workers,
            prefetch_factor=self.args.prefetch_factor,
            pin_memory=True,
            drop_last=True,
            persistent_workers=True,
        )
