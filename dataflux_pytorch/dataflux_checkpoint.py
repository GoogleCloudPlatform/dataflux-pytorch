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

from io import BytesIO
from typing import Optional

from dataflux_core import user_agent
from dataflux_pytorch.multipart_upload.multipart import \
    upload_chunks_concurrently_from_bytesio as upload
from google.cloud import storage
from google.cloud.storage.fileio import BlobReader, BlobWriter


class DatafluxCheckpoint():
    """Implements the interface of saving and loading model checkpoints.

    The reader and writer return a BlobReader and BlobWriter respectively, which
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
                be created during initialization with background authentication.
        """
        self.project_name = project_name
        self.bucket_name = bucket_name
        self.storage_client = storage_client
        if not storage_client:
            self.storage_client = storage.Client(project=self.project_name, )
        user_agent.add_dataflux_user_agent(self.storage_client)
        self.bucket = self.storage_client.bucket(self.bucket_name)

    def reader(self, object_name: str) -> BlobReader:
        blob = self.bucket.blob(object_name)
        stream = BytesIO()
        blob.download_to_file(stream)
        stream.seek(0)
        return stream

    def writer(self, object_name: str) -> BlobWriter:
        blob = self.bucket.blob(object_name)
        return DatafluxCheckpointBuffer(blob)


class DatafluxCheckpointBuffer(BytesIO):
    """Implements a BytesIO buffer that will flush to GCS.

    This class overrides the flush function of BytesIO to perform
    an optimized multipart upload directly to a specified GCS bucket.
    """

    def __init__(self, blob: "google.cloud.storage.blob.Blob"):
        """Initializes the DatafluxCheckpointBuffer

        Args:
            blob: A GCS blob to which the checkpoint will be uploaded.
        """
        self.blob = blob
        super().__init__()

    def flush(self):
        super().flush()
        upload(self, self.blob)
