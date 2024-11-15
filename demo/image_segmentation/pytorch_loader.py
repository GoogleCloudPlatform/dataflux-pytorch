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
import random

import dataflux_core
import numpy as np
import scipy.ndimage
from google.cloud import storage
from torch.utils.data import Dataset
from torchvision import transforms

from dataflux_pytorch import dataflux_mapstyle_dataset


def get_train_transforms():
    rand_flip = RandFlip()
    cast = Cast(types=(np.float32, np.uint8))
    rand_scale = RandomBrightnessAugmentation(factor=0.3, prob=0.1)
    rand_noise = GaussianNoise(mean=0.0, std=0.1, prob=0.1)
    train_transforms = transforms.Compose(
        [rand_flip, cast, rand_scale, rand_noise])
    return train_transforms


class RandBalancedCrop:

    def __init__(self, patch_size, oversampling):
        self.patch_size = patch_size
        self.oversampling = oversampling

    def __call__(self, data):
        image, label = data["image"], data["label"]
        if random.random() < self.oversampling:
            image, label, cords = self.rand_foreg_cropd(image, label)
        else:
            image, label, cords = self._rand_crop(image, label)
        data.update({"image": image, "label": label})
        return data

    @staticmethod
    def randrange(max_range):
        return 0 if max_range == 0 else random.randrange(max_range)

    def get_cords(self, cord, idx):
        return cord[idx], cord[idx] + self.patch_size[idx]

    def _rand_crop(self, image, label):
        ranges = [s - p for s, p in zip(image.shape[1:], self.patch_size)]
        cord = [self.randrange(x) for x in ranges]
        low_x, high_x = self.get_cords(cord, 0)
        low_y, high_y = self.get_cords(cord, 1)
        low_z, high_z = self.get_cords(cord, 2)
        image = image[:, low_x:high_x, low_y:high_y, low_z:high_z]
        label = label[:, low_x:high_x, low_y:high_y, low_z:high_z]
        return image, label, [low_x, high_x, low_y, high_y, low_z, high_z]

    def rand_foreg_cropd(self, image, label):

        def adjust(foreg_slice, patch_size, label, idx):
            diff = patch_size[idx - 1] - (foreg_slice[idx].stop -
                                          foreg_slice[idx].start)
            sign = -1 if diff < 0 else 1
            diff = abs(diff)
            ladj = self.randrange(diff)
            hadj = diff - ladj
            low = max(0, foreg_slice[idx].start - sign * ladj)
            high = min(label.shape[idx], foreg_slice[idx].stop + sign * hadj)
            diff = patch_size[idx - 1] - (high - low)
            if diff > 0 and low == 0:
                high += diff
            elif diff > 0:
                low -= diff
            return low, high

        cl = np.random.choice(np.unique(label[label > 0]))
        foreg_slices = scipy.ndimage.find_objects(
            scipy.ndimage.measurements.label(label == cl)[0])
        foreg_slices = [x for x in foreg_slices if x is not None]
        slice_volumes = [np.prod([s.stop - s.start for s in sl])
                         for sl in foreg_slices]
        slice_idx = np.argsort(slice_volumes)[-2:]
        foreg_slices = [foreg_slices[i] for i in slice_idx]
        if not foreg_slices:
            return self._rand_crop(image, label)
        foreg_slice = foreg_slices[random.randrange(len(foreg_slices))]
        low_x, high_x = adjust(foreg_slice, self.patch_size, label, 1)
        low_y, high_y = adjust(foreg_slice, self.patch_size, label, 2)
        low_z, high_z = adjust(foreg_slice, self.patch_size, label, 3)
        image = image[:, low_x:high_x, low_y:high_y, low_z:high_z]
        label = label[:, low_x:high_x, low_y:high_y, low_z:high_z]
        return image, label, [low_x, high_x, low_y, high_y, low_z, high_z]


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
            self.storage_client = storage.Client(
                project=project_name,
            )

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
                    )
                ),
            )

            label = np.load(
                io.BytesIO(
                    dataflux_core.download.download_single(
                        storage_client=self.storage_client,
                        bucket_name=self.bucket_name,
                        object_name=self.labels[idx][0],
                    )
                ),
            )
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
                img = np.load(io.BytesIO(
                    images_in_bytes[i]), allow_pickle=True)
                lab = np.load(io.BytesIO(
                    labels_in_bytes[i]), allow_pickle=True)
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
