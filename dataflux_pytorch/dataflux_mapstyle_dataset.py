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

import logging
import multiprocessing
import os
import warnings

import dataflux_core
from google.cloud import storage
from google.cloud.storage.retry import DEFAULT_RETRY
from torch.utils import data

from dataflux_pytorch._helper import _get_missing_permissions

MODIFIED_RETRY = DEFAULT_RETRY.with_deadline(100000.0).with_delay(
    initial=1.0, multiplier=1.5, maximum=30.0)

FORK = "fork"
CREATE = "storage.objects.create"
DELETE = "storage.objects.delete"


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

        disable_compose: A boolean flag indicating if compose download should be active.
        Compose should be disabled for highly scaled implementations.

        list_retry_config: A google API retry for Dataflux fast list operations. This allows
        for retry backoff configuration.

        download_retry_config: A google API retry for Dataflux download operations. This allows
        for retry backoff configuration.
    """

    def __init__(
        self,
        sort_listing_results: bool = False,
        max_composite_object_size: int = 100000000,
        num_processes: int = os.cpu_count(),
        prefix: str = None,
        max_listing_retries: int = 3,
        threads_per_process: int = 1,
        disable_compose: bool = False,
        list_retry_config:
        "google.api_core.retry.retry_unary.Retry" = MODIFIED_RETRY,
        download_retry_config:
        "google.api_core.retry.retry_unary.Retry" = MODIFIED_RETRY,
    ):
        self.sort_listing_results = sort_listing_results
        self.max_composite_object_size = max_composite_object_size
        self.num_processes = num_processes
        self.prefix = prefix
        self.max_listing_retries = max_listing_retries
        self.threads_per_process = threads_per_process
        if disable_compose:
            self.max_composite_object_size = 0
        self.list_retry_config = list_retry_config
        self.download_retry_config = download_retry_config


def data_format_default(data):
    return data


class DataFluxMapStyleDataset(data.Dataset):

    def __init__(
        self,
        project_name,
        bucket_name,
        config=Config(),
        data_format_fn=data_format_default,
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
                to access the project and the bucket. If not specified, one will be created
                when needed.

        Returns:
            None.
        """
        super().__init__()
        multiprocessing_start = multiprocessing.get_start_method(
            allow_none=False)
        if storage_client is not None and multiprocessing_start != FORK:
            warnings.warn(
                "Setting the storage client is not fully supported when multiprocessing starts with spawn or forkserver.",
                UserWarning,
            )
        self.storage_client = storage_client
        self.project_name = project_name
        self.bucket_name = bucket_name
        self.data_format_fn = data_format_fn
        self.config = config
        if self.storage_client is None:
            self.storage_client = storage.Client(project=self.project_name)
        # If composed download is enabled and a storage_client was provided,
        # check if the client has permissions to create and delete the
        # composed object.
        if storage_client is not None and self.config.max_composite_object_size:
            missing_perm = _get_missing_permissions(
                storage_client=self.storage_client,
                bucket_name=self.bucket_name,
                project_name=self.project_name,
                required_perm=[CREATE, DELETE])
            if missing_perm and len(missing_perm) > 0:
                raise PermissionError(
                    f"Missing permissions {', '.join(missing_perm)} for composed download. To disable composed download set config.disable_compose=True or to enable composed download, grant missing permissions."
                )

        self.dataflux_download_optimization_params = (
            dataflux_core.download.DataFluxDownloadOptimizationParams(
                max_composite_object_size=self.config.max_composite_object_size
            ))

        self.objects = self._list_GCS_blobs_with_retry()

    def __len__(self):
        return len(self.objects)

    def __getitem__(self, idx):
        return self.data_format_fn(
            dataflux_core.download.download_single(
                storage_client=self.storage_client,
                bucket_name=self.bucket_name,
                object_name=self.objects[idx][0],
                retry_config=self.config.download_retry_config,
            ))

    def __getitems__(self, indices):
        return [
            self.data_format_fn(bytes_content) for bytes_content in
            dataflux_core.download.dataflux_download_threaded(
                project_name=self.project_name,
                bucket_name=self.bucket_name,
                objects=[self.objects[idx] for idx in indices],
                storage_client=self.storage_client,
                dataflux_download_optimization_params=self.
                dataflux_download_optimization_params,
                threads=self.config.threads_per_process,
                retry_config=self.config.download_retry_config,
            )
        ]

    def _list_GCS_blobs_with_retry(self):
        """Retries Dataflux Listing upon exceptions, up to the retries defined in self.config."""
        error = None
        listed_objects = []
        for _ in range(self.config.max_listing_retries):
            try:
                lister = dataflux_core.fast_list.ListingController(
                    max_parallelism=self.config.num_processes,
                    project=self.project_name,
                    bucket=self.bucket_name,
                    sort_results=self.config.sort_listing_results,
                    prefix=self.config.prefix,
                    retry_config=self.config.list_retry_config,
                )
                # If the dataset was not initialized with an storage_client, ensure that we do not attach a client to the lister to avoid pickling errors (#58).
                if self.storage_client is not None:
                    lister.client = self.storage_client
                listed_objects = lister.run()

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

    def _get_or_create_storage_client(self):
        """Provide initialized storage client or construct one.

        Some environments do not support pickling client objects, so
        storage_client is an optional parameter. If the dataset was
        initialized without a storage client, construct and provide one for
        use.
        """
        if self.storage_client is not None:
            return self.storage_client
        return storage.Client(project=self.project_name)

    def __getstate__(self):
        # Copy the object's state from self.__dict__ which contains
        # all our instance attributes. Use the dict.copy()
        # method to avoid modifying the original state.
        state = self.__dict__.copy()
        # Remove the unpicklable entries.
        del state['storage_client']
        return state

    def __setstate__(self, state):
        # Restore instance attributes.
        self.__dict__.update(state)
        # Restore the storage client with initial params.
        self.storage_client = storage.Client(project=self.project_name)
