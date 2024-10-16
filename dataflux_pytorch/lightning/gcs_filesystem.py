import io
import os
from contextlib import contextmanager
from pathlib import Path
from typing import Generator, Optional, Union

import torch.distributed as dist
from dataflux_core import user_agent
from dataflux_pytorch.dataflux_checkpoint import DatafluxCheckpointBuffer
from dataflux_pytorch.lightning.path_utils import parse_gcs_path
from google.cloud import storage
from torch.distributed.checkpoint import FileSystemReader, FileSystemWriter


class GCSFileSystem():

    def __init__(
        self,
        project_name: str,
        debug: bool,
        storage_client: Optional[storage.Client] = None,
    ):
        self.project_name = project_name
        self.storage_client = storage_client
        if not storage_client:
            self.storage_client = storage.Client(project=self.project_name)
        user_agent.add_dataflux_user_agent(self.storage_client)
        self.debug = debug

    @contextmanager
    def create_stream(self, path: Union[str, os.PathLike],
                      mode: str) -> Generator[io.IOBase, None, None]:
        bucket, path = parse_gcs_path(path)
        blob = self.storage_client.bucket(bucket).blob(path)
        if mode == "wb":  # write mode.
            if self.debug:
                print(
                    f'Writing from Rank: {dist.get_rank()} to bucket {bucket}\
                        &path {path} ...')
            with DatafluxCheckpointBuffer(blob) as stream:
                yield stream
        elif mode == "rb":  # read mode.
            if self.debug:
                print(
                    f'Reading on Rank: {dist.get_rank()} from bucket {bucket} &\
                        path {path} ...')
            stream = io.BytesIO()
            blob.download_to_file(stream)
            stream.seek(0)
            yield stream
        else:
            raise ValueError(
                "Invalid mode argument, create_stream only supports rb (read mode) & wb (write mode)"
            )

    def concat_path(self, path: Union[str, os.PathLike],
                    suffix: str) -> Union[str, os.PathLike]:
        if not isinstance(path, Path):
            path = Path(path)
        return path / suffix

    def init_path(self, path: Union[str,
                                    os.PathLike]) -> Union[str, os.PathLike]:
        if not isinstance(path, Path):
            path = Path(path)
        return path

    def rename(self, path: Union[str, os.PathLike],
               new_path: Union[str, os.PathLike]) -> None:
        old_bucket, old_path = parse_gcs_path(path)
        new_bucket, new_path = parse_gcs_path(new_path)
        if old_bucket != new_bucket:
            raise Exception(f"When renaming objects, the old bucket name \
                    (got: {old_bucket}) must be the same as the new bucket\
                    name (got: {new_bucket})")
        blob = self.storage_client.bucket(old_bucket).blob(old_path)
        self.storage_client.bucket(new_bucket).rename_blob(blob, new_path)

    def mkdir(self, path: Union[str, os.PathLike]) -> None:
        pass

    def exists(self, path: Union[str, os.PathLike]) -> bool:
        bucket, path = parse_gcs_path(path)
        blob = self.storage_client.bucket(bucket).blob(path)
        return blob.exists()

    def rm_file(self, path: Union[str, os.PathLike]) -> None:
        bucket, path = parse_gcs_path(path)
        blob = self.storage_client.bucket(bucket).blob(path)
        blob.delete()

    @classmethod
    def validate_checkpoint_id(cls, checkpoint_id: Union[str,
                                                         os.PathLike]) -> bool:
        if isinstance(checkpoint_id, Path):
            return True
        # parse_gcs_path will raise exception if path is not in valid.
        parse_gcs_path(checkpoint_id)
        return True


class GCSDistributedWriter(FileSystemWriter):

    def __init__(self,
                 path,
                 project_name: str,
                 storage_client: Optional[storage.Client] = None,
                 debug: Optional[bool] = False,
                 **kwargs):
        super().__init__(path, **kwargs)
        self.fs = GCSFileSystem(project_name, storage_client, debug)
        self.sync_files = False


class GCSDistributedReader(FileSystemReader):

    def __init__(self,
                 path: Union[str, os.PathLike],
                 project_name: str,
                 storage_client: Optional[storage.Client] = None,
                 debug: Optional[bool] = False,
                 **kwargs):
        super().__init__(path, **kwargs)
        self.fs = GCSFileSystem(project_name, storage_client, debug)
