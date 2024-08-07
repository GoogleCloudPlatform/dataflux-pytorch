from typing import Any, Dict, Optional

import torch
from dataflux_core import user_agent
from google.cloud import storage
from lightning.pytorch.plugins.io import CheckpointIO


class DatafluxLightningCheckpoint(CheckpointIO):
    """A checkpoint manager for GCS using the :class:'CheckpointIO' interface"""

    def __init__(
        self,
        project_name: str,
        bucket_name: str,
        storage_client: Optional[storage.Client] = None,
    ):
        self.project_name = project_name
        self.bucket_name = bucket_name
        self.storage_client = storage_client
        if not storage_client:
            self.storage_client = storage.Client(project=self.project_name, )
        user_agent.add_dataflux_user_agent(self.storage_client)
        self.bucket = self.storage_client.bucket(self.bucket_name)

    def _parse_gcs_path(self, path: str) -> str:
        path = str(path)
        sep = "//"
        if not path or not (path.startswith("gcs://")
                            or path.startswith("gs://")
                            or path.startswith("gs:/")):
            raise ValueError(
                f"Path needs to begin with gcs:// or gs://, got {path}")
        if not path.startswith("gs://") and path.startswith("gs:/"):
            sep = "/"
        path = path.split(sep, maxsplit=1)
        if not path or len(path) < 2:
            raise ValueError("Bucket name must be non-empty")
        split = path[1].split("/", maxsplit=1)
        if len(split) == 1:
            bucket = split[0]
            prefix = ""
        else:
            bucket, prefix = split
        if not bucket:
            raise ValueError("Bucket name must be non-empty")
        if bucket != self.bucket_name:
            raise ValueError(
                f'Unexpected bucket name, expected {self.bucket_name} got {bucket}'
            )
        return prefix

    def save_checkpoint(
        self,
        checkpoint: Dict[str, Any],
        path: str,
        storage_options: Optional[Any] = None,
    ) -> None:
        key = self._parse_gcs_path(path)
        blob = self.bucket.blob(key)
        with blob.open("wb", ignore_flush=True) as blobwriter:
            torch.save(checkpoint, blobwriter)

    def load_checkpoint(
        self,
        path: str,
        map_location: Optional[Any] = None,
    ) -> Dict[str, Any]:
        key = self._parse_gcs_path(path)
        blob = self.bucket.blob(key)
        return torch.load(blob.open("rb"), map_location)

    def remove_checkpoint(
        self,
        path: str,
    ) -> None:
        key = self._parse_gcs_path(path)
        blob = self.bucket.blob(key)
        blob.delete()

    def teardown(self, ) -> None:
        pass
