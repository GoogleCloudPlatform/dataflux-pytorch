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
from pathlib import Path
from typing import Any, Dict, Optional, Tuple, Union

import torch
import os
from dataflux_core import user_agent
from google.cloud import storage
from lightning.pytorch.plugins.io import CheckpointIO
from dataflux_pytorch.lightning.path_utils import parse_gcs_path
from dataflux_pytorch.multipart_upload.multipart import upload_chunks_concurrently_from_bytesio as upload


class DatafluxLightningCheckpoint(CheckpointIO):
    """A checkpoint manager for GCS using the :class:'CheckpointIO' interface"""

    def __init__(
        self,
        project_name: str,
        storage_client: Optional[storage.Client] = None,
        enable_multipart: bool = False,
    ):
        self.project_name = project_name
        self.storage_client = storage_client
        self.enable_multipart = enable_multipart
        if not storage_client:
            self.storage_client = storage.Client(project=self.project_name, )
        user_agent.add_dataflux_user_agent(self.storage_client)

    def save_checkpoint(
        self,
        checkpoint: Dict[str, Any],
        path: Union[str, Path],
        storage_options: Optional[Any] = None,
    ) -> None:
        bucket_name, key = parse_gcs_path(path)
        bucket_client = self.storage_client.bucket(bucket_name)
        blob = bucket_client.blob(key)
        if self.enable_multipart:
            fb = io.BytesIO()
            torch.save(checkpoint, fb)
            upload(fb, blob)
        else:
            print(
                f"\n### From GCS Save checkpoint ### saving for key {key} and bucket name: {bucket_name} ###\n")
            with blob.open("wb", ignore_flush=True) as blobwriter:
                torch.save(checkpoint, blobwriter)

    def load_checkpoint(
        self,
        path: Union[str, Path],
        map_location: Optional[Any] = None,
    ) -> Dict[str, Any]:
        bucket_name, key = parse_gcs_path(path)
        bucket_client = self.storage_client.bucket(bucket_name)
        blob = bucket_client.blob(key)
        stream = io.BytesIO()
        blob.download_to_file(stream)
        stream.seek(0)
        return torch.load(stream, map_location)

    def remove_checkpoint(
        self,
        path: Union[str, Path],
    ) -> None:
        bucket_name, key = parse_gcs_path(path)
        bucket_client = self.storage_client.bucket(bucket_name)
        blob = bucket_client.blob(key)
        blob.delete()

    def teardown(self, ) -> None:
        pass
