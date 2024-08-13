from lightning.pytorch.plugins.io import CheckpointIO
from google.cloud import storage
from dataflux_core import user_agent
from pathlib import Path
from typing import Any, Dict, Optional, Union, Tuple

import torch
import os
import time
import os
import time
from dataflux_core import user_agent
from google.cloud import storage
from lightning.pytorch.plugins.io import CheckpointIO


class DatafluxLightningCheckpoint(CheckpointIO):
    """A checkpoint manager for GCS using the :class:'CheckpointIO' interface"""

    def __init__(
        self,
        project_name: str,
        storage_client: Optional[storage.Client] = None,
    ):
        self.project_name = project_name
        self.storage_client = storage_client
        if not storage_client:
            self.storage_client = storage.Client(project=self.project_name, )
        user_agent.add_dataflux_user_agent(self.storage_client)

    def _process_input_path(self, path: Union[str, Path]) -> str:
        if isinstance(path, str):
            return path
        elif isinstance(path, Path):
            # When casting from Path object to string, it considers cloud URLs as Network URLs and gets rid of //
            scheme, rest = str(path).split(":/")
            return str(scheme) + "://" + str(rest)
        else:
            raise TypeError(
                "path argument must be of type string or pathlib.Path object")

    def _parse_gcs_path(self, path: Union[str, Path]) -> Tuple[str, str]:
        if not path:
            raise ValueError("Path cannot be empty")
        input_path = self._process_input_path(path)
        if not (input_path.startswith("gcs://")
                or input_path.startswith("gs://")):
            raise ValueError("Path needs to begin with gcs:// or gs://")
        input_path = input_path.split("//", maxsplit=1)
        if not input_path or len(input_path) < 2:
            raise ValueError("Bucket name must be non-empty")
        split = input_path[1].split("/", maxsplit=1)
        bucket_name = ""
        if len(split) == 1:
            bucket_name = split[0]
            prefix = ""
        else:
            bucket_name, prefix = split
        if not bucket_name:
            raise ValueError("Bucket name must be non-empty")
        return bucket_name, prefix

    def save_checkpoint(
        self,
        checkpoint: Dict[str, Any],
        path: Union[str, Path],
        storage_options: Optional[Any] = None,
    ) -> None:
        bucket_name, key = self._parse_gcs_path(path)
        rest_of_the_path, _ = os.path.split(key)
        os.makedirs(os.path.dirname(rest_of_the_path+"/"), exist_ok=True)
        with open(key, 'wb') as f:
            torch.save(checkpoint, f)

        bucket_client = self.storage_client.bucket(bucket_name)

        blob = bucket_client.blob(key)
        try:
            with open(key, 'rb') as f:  # Open in read-binary mode
                blob.upload_from_file(f)
        except Exception as e:
            print(f"Error uploading to GCS: {e}")

    def load_checkpoint(
        self,
        path: Union[str, Path],
        map_location: Optional[Any] = None,
    ) -> Dict[str, Any]:
        bucket_name, key = self._parse_gcs_path(path)
        bucket_client = self.storage_client.bucket(bucket_name)
        blob = bucket_client.blob(key)
        return torch.load(blob.open("rb"), map_location)

    def remove_checkpoint(
        self,
        path: Union[str, Path],
    ) -> None:
        bucket_name, key = self._parse_gcs_path(path)
        bucket_client = self.storage_client.bucket(bucket_name)
        blob = bucket_client.blob(key)
        blob.delete()

    def teardown(self, ) -> None:
        pass
