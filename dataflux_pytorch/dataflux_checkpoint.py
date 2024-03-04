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

from google.cloud import storage
from google.cloud.storage.fileio import BlobReader, BlobWriter
from google.api_core.client_info import ClientInfo

from typing import Optional


class DatafluxCheckpoint:
    """A class that implements the interface of saving and loading model checkpoints.

    The reader and writer returns a BlobReader and BlobWriter respectively, which
    both implement io.BufferedIOBase. Therefore, they can be safely passed to torch.load()
    and torch.save() to load and save model checkpoints.
    """

    def __init__(
        self,
        project_name: str,
        bucket_name: str,
        storage_client: Optional[storage.Client] = None,
    ):
        """Initializes the DatafluxCheckpoint.

        Args:
            project_name: The name of the GCP project.
            bucket_name: The name of the GCS bucket that is going to hold the checkpoint.
            storage_client: The google.cloud.storage.Client object initiated with sufficient
                permission to access the project and the bucket. If not specified, it will
                be created during initialization.

        Returns:
            None.
        """
        self.project_name = project_name
        self.bucket_name = bucket_name
        self.storage_client = storage_client
        if not storage_client:
            self.storage_client = storage.Client(
                project=self.project_name,
                client_info=ClientInfo(user_agent="dataflux/0.0"),
            )
        self.bucket = self.storage_client.bucket(self.bucket_name)

    def reader(self, object_name: str) -> BlobReader:
        blob = self.bucket.blob(object_name)
        return blob.open("rb")

    def writer(self, object_name: str) -> BlobWriter:
        blob = self.bucket.blob(object_name)
        return blob.open("wb", ignore_flush=True)
