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

import dataflux_core
import numpy as np
from google.cloud import storage
from torch.utils.data import Dataset

from dataflux_pytorch import dataflux_mapstyle_dataset
from demo.shared.image_segmentation import RandBalancedCrop, get_train_transforms


class PytTrain(Dataset):

    def __init__(self, images, labels, **kwargs):
        self.images, self.labels = images, labels
        self.train_transforms = get_train_transforms()
        patch_size, oversampling = kwargs["patch_size"], kwargs["oversampling"]
        self.patch_size = patch_size
        self.rand_crop = RandBalancedCrop(patch_size=patch_size,
                                          oversampling=oversampling)

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        try:
            image = np.load(self.images[idx])
            label = np.load(self.labels[idx])
        except:
            image = None
            label = None
        data = {"image": image, "label": label}
        if data["image"] is not None and data["label"] != None:
            data = self.rand_crop(data)
            data = self.train_transforms(data)
        return data["image"], data["label"]


class DatafluxPytTrain(Dataset):

    def __init__(
        self,
        project_name,
        bucket_name,
        config=dataflux_mapstyle_dataset.Config(),
        storage_client=None,
        **kwargs,
    ):
        # Data transformation setup.
        self.train_transforms = get_train_transforms()
        patch_size, oversampling = kwargs["patch_size"], kwargs["oversampling"]
        images_prefix, labels_prefix = kwargs["images_prefix"], kwargs[
            "labels_prefix"]
        self.patch_size = patch_size
        self.rand_crop = RandBalancedCrop(patch_size=patch_size,
                                          oversampling=oversampling)

        # Dataflux-specific setup.
        self.project_name = project_name
        self.bucket_name = bucket_name
        self.config = config
        self.dataflux_download_optimization_params = (
            dataflux_core.download.DataFluxDownloadOptimizationParams(
                max_composite_object_size=self.config.max_composite_object_size
            ))
        if not storage_client:
            self.storage_client = storage.Client(project=project_name, )

        # Data listing.
        self.images = dataflux_core.fast_list.ListingController(
            max_parallelism=self.config.num_processes,
            project=self.project_name,
            bucket=self.bucket_name,
            # This needs to be True to map images with labels.
            sort_results=self.config.sort_listing_results,
            prefix=images_prefix,
        ).run()
        self.labels = dataflux_core.fast_list.ListingController(
            max_parallelism=self.config.num_processes,
            project=self.project_name,
            bucket=self.bucket_name,
            # This needs to be True to map images with labels.
            sort_results=self.config.sort_listing_results,
            prefix=labels_prefix,
        ).run()

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        try:
            image = np.load(
                io.BytesIO(
                    dataflux_core.download.download_single(
                        storage_client=self.storage_client,
                        bucket_name=self.bucket_name,
                        object_name=self.images[idx][0],
                    )), )

            label = np.load(
                io.BytesIO(
                    dataflux_core.download.download_single(
                        storage_client=self.storage_client,
                        bucket_name=self.bucket_name,
                        object_name=self.labels[idx][0],
                    )), )
        except:
            image = None
            label = None

        data = {"image": image, "label": label}
        if data["image"] is not None and data["label"] is not None:
            data = self.rand_crop(data)
            data = self.train_transforms(data)
        return data["image"], data["label"]

    def __getitems__(self, indices):
        images_in_bytes = dataflux_core.download.dataflux_download(
            project_name=self.project_name,
            bucket_name=self.bucket_name,
            objects=[self.images[idx] for idx in indices],
            storage_client=self.storage_client,
            dataflux_download_optimization_params=self.
            dataflux_download_optimization_params,
        )

        labels_in_bytes = dataflux_core.download.dataflux_download(
            project_name=self.project_name,
            bucket_name=self.bucket_name,
            objects=[self.labels[idx] for idx in indices],
            storage_client=self.storage_client,
            dataflux_download_optimization_params=self.
            dataflux_download_optimization_params,
        )

        res = []
        for i in range(len(images_in_bytes)):
            try:
                img = np.load(io.BytesIO(images_in_bytes[i]),
                              allow_pickle=True)
                lab = np.load(io.BytesIO(labels_in_bytes[i]),
                              allow_pickle=True)
            except:
                img = None
                lab = None
            data = {
                "image": img,
                "label": lab,
            }
            if data["image"] is not None and data["label"] is not None:
                data = self.rand_crop(data)
                data = self.train_transforms(data)
            res.append((data["image"], data["label"]))
        return res


class PytVal(Dataset):

    def __init__(self, images, labels):
        self.images, self.labels = images, labels

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        try:
            image = np.load(self.images[idx])
            label = np.load(self.labels[idx])
        except:
            image = None
            label = None
        return image, label
