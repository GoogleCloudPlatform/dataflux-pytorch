import io
import os
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Dict, Generator, Optional, Tuple, Union, cast

import torch
from dataflux_core import user_agent
from google.cloud import storage
from lightning.pytorch.plugins.io import CheckpointIO
from torch.distributed.checkpoint import FileSystemWriter
from torch.distributed.checkpoint.filesystem import FileSystem, FileSystemBase


def _process_input_path(path: Union[str, Path]) -> str:
    if isinstance(path, str):
        return path
    elif isinstance(path, Path):
        # When casting from Path object to string, it considers cloud URLs as Network URLs and gets rid of //
        scheme, rest = str(path).split(":/")
        return str(scheme) + "://" + str(rest)
    else:
        raise TypeError(
            "path argument must be of type string or pathlib.Path object")


def _parse_gcs_path(path: Union[str, Path]) -> Tuple[str, str]:
    if not path:
        raise ValueError("Path cannot be empty")
    input_path = _process_input_path(path)
    if not (input_path.startswith("gcs://") or input_path.startswith("gs://")):
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

    def save_checkpoint(
        self,
        checkpoint: Dict[str, Any],
        path: Union[str, Path],
        storage_options: Optional[Any] = None,
    ) -> None:
        bucket_name, key = _parse_gcs_path(path)
        bucket_client = self.storage_client.bucket(bucket_name)
        blob = bucket_client.blob(key)
        with blob.open("wb", ignore_flush=True) as blobwriter:
            torch.save(checkpoint, blobwriter)

    def load_checkpoint(
        self,
        path: Union[str, Path],
        map_location: Optional[Any] = None,
    ) -> Dict[str, Any]:
        bucket_name, key = _parse_gcs_path(path)
        bucket_client = self.storage_client.bucket(bucket_name)
        blob = bucket_client.blob(key)
        return torch.load(blob.open("rb"), map_location)

    def remove_checkpoint(
        self,
        path: Union[str, Path],
    ) -> None:
        bucket_name, key = _parse_gcs_path(path)
        bucket_client = self.storage_client.bucket(bucket_name)
        blob = bucket_client.blob(key)
        blob.delete()

    def teardown(self, ) -> None:
        pass


class GCSFileSystem(FileSystemBase):

    def __init__(
        self,
        project_name: str,
        storage_client: Optional[storage.Client] = None,
    ):
        self.project_name = project_name
        self.storage_client = storage_client
        if not storage_client:
            self.storage_client = storage.Client(project=self.project_name)
        user_agent.add_dataflux_user_agent(self.storage_client)

    @contextmanager
    def create_stream(self, path: Union[str, os.PathLike],
                      mode: str) -> Generator[io.IOBase, None, None]:
        bucket, path = _parse_gcs_path(path)
        with self.storage_client.bucket(bucket).blob(path).open(
                "wb", ignore_flush=True) as stream:
            yield cast(io.IOBase, stream)

    def concat_path(self, path: Union[str, os.PathLike],
                    suffix: str) -> Union[str, os.PathLike]:
        return cast(Path, path) / suffix

    def init_path(self, path: Union[str,
                                    os.PathLike]) -> Union[str, os.PathLike]:
        if not isinstance(path, Path):
            path = Path(path)
        return path

    def rename(self, path: Union[str, os.PathLike],
               new_path: Union[str, os.PathLike]) -> None:
        old_bucket, old_path = _parse_gcs_path(path)
        new_bucket, new_path = _parse_gcs_path(new_path)
        if old_bucket != new_bucket:
            raise Exception(
                f"When renaming objects, the old bucket name (got: {old_bucket}) must be the same as the new bucket name (got: {new_bucket})"
            )
        blob = self.storage_client.bucket(old_bucket).blob(old_path)
        self.storage_client.bucket(new_bucket).rename_blob(blob, new_path)

    def mkdir(self, path: Union[str, os.PathLike]) -> None:
        pass

    @classmethod
    def validate_checkpoint_id(cls, checkpoint_id: Union[str,
                                                         os.PathLike]) -> bool:
        if isinstance(checkpoint_id, Path):
            return True

        _parse_gcs_path(checkpoint_id)
        return True


class GCSDistributedWriter(FileSystemWriter):

    def __init__(self,
                 path,
                 project_name: str,
                 storage_client: Optional[storage.Client] = None,
                 **kwargs):
        super().__init__(path, **kwargs)
        self.fs = GCSFileSystem(project_name, storage_client)
        self.sync_files = False
