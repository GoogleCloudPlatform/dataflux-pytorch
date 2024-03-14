"""
 Copyright 2023 Google LLC

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
import logging

from torch.utils import data
from google.cloud import storage
from google.api_core.client_info import ClientInfo

import dataflux_core


class Config:
    """Customizable configuration to the DataFluxMapStyleDataset.

    Attributes:
        sort_listing_results: A boolean flag indicating if data listing results
            will be alphabetically sorted. Default to False.

        max_composite_object_size: An integer indicating a cap for the maximum
            size of the composite object in bytes. Default to 100000000 = 100 MiB.

        num_processes: The number of processes to be used in the Dataflux algorithms.
            Default to the number of CPUs from the running environment.

        prefix: The prefix that is used to list the objects in the bucket with.
            The default is None which means it will list all the objects in the bucket.

        max_listing_retries: An integer indicating the maximum number of retries
        to attempt in case of any Python multiprocessing errors during
        GCS objects listing. Default to 3.
    """

    def __init__(
        self,
        sort_listing_results: bool = False,
        max_composite_object_size: int = 100000000,
        num_processes: int = os.cpu_count(),
        prefix: str = None,
        max_listing_retries: int = 3,
    ):
        self.sort_listing_results = sort_listing_results
        self.max_composite_object_size = max_composite_object_size
        self.num_processes = num_processes
        self.prefix = prefix
        self.max_listing_retries = max_listing_retries


class DataFluxMapStyleDataset(data.Dataset):
    def __init__(
        self,
        project_name,
        bucket_name,
        config=Config(),
        data_format_fn=lambda data: data,
        storage_client=None,
    ):
        """Initializes the DataFluxMapStyleDataset.

        The initialization sets up the needed configuration and runs data
        listing using the Dataflux algorithm.

        Args:
            project_name: The name of the GCP project.
            bucket_name: The name of the GCS bucket that holds the objects to compose.
                The Dataflux download algorithm uploads the the composed object to this bucket too.
            destination_blob_name: The name of the composite object to be created.
            config: A dataflux_mapstyle_dataset.Config object that includes configuration
                customizations. If not specified, a default config with default parameters is created.
            data_format_fn: A function that formats the downloaded bytes to the desired format.
                If not specified, the default formatting function leaves the data as-is.
            storage_client: The google.cloud.storage.Client object initiated with sufficient permission
                to access the project and the bucket. If not specified, it will be created
                during initialization.

        Returns:
            None.
        """
        super().__init__()
        self.storage_client = storage_client
        if not storage_client:
            self.storage_client = storage.Client(
                project=project_name,
                client_info=ClientInfo(user_agent="dataflux/0.0"),
            )
        self.project_name = project_name
        self.bucket_name = bucket_name
        self.data_format_fn = data_format_fn
        self.config = config
        self.dataflux_download_optimization_params = (
            dataflux_core.download.DataFluxDownloadOptimizationParams(
                max_composite_object_size=self.config.max_composite_object_size
            )
        )

        self.objects = self._list_GCS_blobs_with_retry()

    def __len__(self):
        return len(self.objects)

    def __getitem__(self, idx):
        return self.data_format_fn(
            dataflux_core.download.download_single(
                storage_client=self.storage_client,
                bucket_name=self.bucket_name,
                object_name=self.objects[idx][0],
            )
        )

    def __getitems__(self, indices):
        return [
            self.data_format_fn(bytes_content)
            for bytes_content in dataflux_core.download.dataflux_download(
                project_name=self.project_name,
                bucket_name=self.bucket_name,
                objects=[self.objects[idx] for idx in indices],
                storage_client=self.storage_client,
                dataflux_download_optimization_params=self.dataflux_download_optimization_params,
            )
        ]

    def _list_GCS_blobs_with_retry(self):
        """Retries Dataflux Listing upon exceptions, up to the retries defined in self.config."""
        error = None
        listed_objects = []
        for _ in range(self.config.max_listing_retries):
            try:
                listed_objects = dataflux_core.fast_list.ListingController(
                    max_parallelism=self.config.num_processes,
                    project=self.project_name,
                    bucket=self.bucket_name,
                    sort_results=self.config.sort_listing_results,
                    prefix=self.config.prefix,
                ).run()
            except Exception as e:
                logging.error(
                    f"exception {str(e)} caught running Dataflux fast listing."
                )
                error = e
                continue

            # No exception -- we can immediately return the listed objects.
            else:
                return listed_objects

        # Did not break the for loop, therefore all attempts
        # raised an exception.
        else:
            raise error
