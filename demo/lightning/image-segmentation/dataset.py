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

from dataflux_pytorch import dataflux_mapstyle_dataset


class DatafluxPytTrain(Dataset):

    def __init__(
        self,
        project_name,
        bucket_name,
        storage_client=None,
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

        self.images_in_bytes = dataflux_mapstyle_dataset.DataFluxMapStyleDataset(
            project_name=self.project_name,
            bucket_name=self.bucket_name,
            config=dataflux_mapstyle_dataset.Config(
                sort_listing_results=True,
                prefix=kwargs["images_prefix"],
            ),
        )

        self.labels_in_bytes = dataflux_mapstyle_dataset.DataFluxMapStyleDataset(
            project_name=self.project_name,
            bucket_name=self.bucket_name,
            config=dataflux_mapstyle_dataset.Config(
                sort_listing_results=True,
                prefix=kwargs["labels_prefix"],
            ),
        )

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        image = np.load(
            io.BytesIO(
                self.images_in_bytes[idx],
            ),
        )

        label = np.load(
            io.BytesIO(
                self.labels_in_bytes[idx],
            ),
        )

        data = {"image": image, "label": label}
        data = self.rand_crop(data)
        data = self.train_transforms(data)
        return data["image"], data["label"]

    def __getitems__(self, indices):
        images_in_bytes_batch = self.images_in_bytes[indices[0]:indices[-1]+1]

        labels_in_bytes_batch = self.labels_in_bytes[indices[0]:indices[-1]+1]

        res = []
        for i in range(len(self.images_in_bytes_batch)):
            data = {
                "image": np.load(io.BytesIO(images_in_bytes_batch[i])),
                "label": np.load(io.BytesIO(labels_in_bytes_batch[i])),
            }
            data = self.rand_crop(data)
            data = self.train_transforms(data)
            res.append((data["image"], data["label"]))
        return res


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
        slice_volumes = [
            np.prod([s.stop - s.start for s in sl]) for sl in foreg_slices
        ]
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
