import torch
from typing import Optional, Dict, Any

from lightning.pytorch.plugins.io import CheckpointIO
from google.cloud import storage
from google.api_core.client_info import ClientInfo

class DatafluxLightningCheckpoint(CheckpointIO):
    """A checkpoint manager for GCS using the :ckass:'CheckpointIO' interface"""

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

    def save_checkpoint(
        self,
        checkpoint: Dict[str, Any],
        path: str,
        storage_options: Optional[Any] = None,
    ) -> None:
        blob = self.bucket.blob(path)
        return torch.save(checkpoint, blob.open("wb", ignore_flush=True))

    def load_checkpoint(
        self,
        path: str,
        map_location: Optional[Any] = None,
    ) -> Dict[str, Any]:
        blob = self.bucket.blob(path)
        return torch.load(blob.open("rb"), map_location)

    def remove_checkpoint(
        self,
        path: str,
    ) -> None:
       """This command may take a while if it is many objects
        https://cloud.google.com/storage/docs/deleting-objects#client-libraries"""
       blob = self.bucket.blob(path)
       blob.delete()

    def teardown(
        self,
    ) -> None:
        pass
