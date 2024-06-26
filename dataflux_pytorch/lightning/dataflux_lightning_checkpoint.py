import torch
from typing import Optional, Dict, Any

from lightning.pytorch.plugins.io import CheckpointIO
from google.cloud import storage
from google.api_core.client_info import ClientInfo

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
            self.storage_client = storage.Client(
                project=self.project_name,
                client_info=ClientInfo(user_agent="dataflux/0.0"),
            )
        self.bucket = self.storage_client.bucket(self.bucket_name)

    def _parse_gcs_path(self, path: str) -> str:
        if not path or not (path.startswith("gcs://") or path.startswith("gs://")):
            raise ValueError("Path needs to begin with gcs:// or gs://")
        path =  path.split("//", maxsplit=1)
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
            raise ValueError(f'Unexpected bucket name, expected {self.bucket_name} got {bucket}')
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
          torch.save(checkpoint, blob.open("wb", ignore_flush=True))

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

    def teardown(
        self,
    ) -> None:
        pass
