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

import io
import numpy as np
from torch.utils.data import Dataset

from dataflux_pytorch import dataflux_mapstyle_dataset
from demo.shared.image_segmentation import RandBalancedCrop


class DatafluxPytTrain(Dataset):

    def __init__(
        self,
        project_name,
        bucket_name,
        **kwargs,
    ):

        self.train_transforms = kwargs["transforms"]
        patch_size, oversampling = kwargs["patch_size"], kwargs["oversampling"]
        self.patch_size = patch_size
        self.rand_crop = RandBalancedCrop(patch_size=patch_size,
                                          oversampling=oversampling)

        # Dataflux-specific setup.
        self.project_name = project_name
        self.bucket_name = bucket_name

        self.images_dataset = dataflux_mapstyle_dataset.DataFluxMapStyleDataset(
            project_name=self.project_name,
            bucket_name=self.bucket_name,
            config=dataflux_mapstyle_dataset.Config(
                # This needs to be True to map images with labels
                sort_listing_results=True,
                prefix=kwargs["images_prefix"],
            ),
        )

        self.labels_dataset = dataflux_mapstyle_dataset.DataFluxMapStyleDataset(
            project_name=self.project_name,
            bucket_name=self.bucket_name,
            config=dataflux_mapstyle_dataset.Config(
                # This needs to be True to map images with labels
                sort_listing_results=True,
                prefix=kwargs["labels_prefix"],
            ),
        )

    def __len__(self):
        return len(self.images_dataset)

    def __getitem__(self, idx):
        image = np.load(io.BytesIO(self.images_dataset[idx], ), )

        label = np.load(io.BytesIO(self.labels_dataset[idx], ), )

        data = {"image": image, "label": label}
        data = self.rand_crop(data)
        data = self.train_transforms(data)
        return data["image"], data["label"]

    def __getitems__(self, indices):
        images_in_bytes_batch = self.images_dataset.__getitems__(indices)
        labels_in_bytes_batch = self.labels_dataset.__getitems__(indices)
        res = []
        for i in range(len(images_in_bytes_batch)):
            data = {
                "image": np.load(io.BytesIO(images_in_bytes_batch[i])),
                "label": np.load(io.BytesIO(labels_in_bytes_batch[i])),
            }
            data = self.rand_crop(data)
            data = self.train_transforms(data)
            res.append((data["image"], data["label"]))
        return res
