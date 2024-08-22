import concurrent.futures
import io
import tempfile
from pathlib import Path
from typing import Any, Dict, Optional, Tuple, Union

import torch
from dataflux_core import user_agent
from google.cloud import storage
from google.cloud.storage import transfer_manager
from lightning.pytorch.plugins.io import CheckpointIO


class DatafluxLightningCheckpoint(CheckpointIO):
    """A checkpoint manager for GCS using the :class:'CheckpointIO' interface"""

    def __init__(
        self,
        project_name: str,
        storage_client: Optional[storage.Client] = None,
        use_transfer_manager: bool = True,
    ):
        self.project_name = project_name
        self.storage_client = storage_client
        self.use_transfer_manager = use_transfer_manager
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
        bucket_client = self.storage_client.bucket(bucket_name)
        blob = bucket_client.blob(key)
        if self.use_transfer_manager:
            # with tempfile.NamedTemporaryFile(mode="w+b", delete=False) as file:
            #     torch.save(checkpoint, file)
            #     transfer_manager.upload_chunks_concurrently(
            #         filename=file.name, blob=blob, worker_type='thread')
            #     return
            stream = io.BytesIO()
            torch.save(checkpoint, stream)
            size = stream.tell()
            stream.seek(0)
            with concurrent.futures.ThreadPoolExecutor(
                    max_workers=8) as executor:
                futures = []
                chunk_size = size // 31  # Produce at most 32 chunks
                if chunk_size < 33554432:
                    chunk_size = 33554432
                i = 0
                while True:
                    data = stream.read(chunk_size)
                    if len(data) == 0:
                        # Reached the end
                        break
                    futures.append(
                        executor.submit(
                            _upload_chunk,
                            bucket_client,
                            key,
                            i,
                            data,
                        ))
                    i += 1
                concurrent.futures.wait(
                    futures,
                    timeout=None,
                    return_when=concurrent.futures.ALL_COMPLETED)
                to_compose_names = []
                for future in futures:
                    to_compose_names.append(future.result())
                to_compose_blobs = [
                    bucket_client.blob(name) for name in to_compose_names
                ]
                blob.compose(to_compose_blobs)
                for blob in to_compose_blobs:
                    blob.delete()
                return

        with blob.open("wb", ignore_flush=True) as blobwriter:
            torch.save(checkpoint, blobwriter)

    def load_checkpoint(
        self,
        path: Union[str, Path],
        map_location: Optional[Any] = None,
    ) -> Dict[str, Any]:
        bucket_name, key = self._parse_gcs_path(path)
        bucket_client = self.storage_client.bucket(bucket_name)
        blob = bucket_client.blob(key)
        if self.use_transfer_manager:
            blob.reload()
            print(blob.size)
            with concurrent.futures.ThreadPoolExecutor(
                    max_workers=8) as executor:
                futures = []
                chunk_size = 67108864
                cursor = 0
                end = blob.size
                while cursor < end:
                    start = cursor
                    cursor = min(cursor + chunk_size, end)
                    futures.append(
                        executor.submit(
                            _download_and_write_chunk_in_place,
                            blob,
                            start=start,
                            end=cursor - 1,
                        ))

                concurrent.futures.wait(
                    futures,
                    timeout=None,
                    return_when=concurrent.futures.ALL_COMPLETED)

            # Raise any exceptions; combine checksums.
            stream = io.BytesIO()
            for future in futures:
                stream.write(future.result())
            stream.seek(0)
            return torch.load(stream, map_location)
        stream = io.BytesIO()
        blob.download_to_file(stream)
        stream.seek(0)
        return torch.load(stream, map_location)

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


def _download_and_write_chunk_in_place(blob: storage.Blob, start: int,
                                       end: int):
    return blob.download_as_bytes(start=start, end=end)


def _upload_chunk(bucket: storage.Bucket, name: str, number: int, data: bytes):
    file_name = name + "." + str(number)
    bucket.blob(file_name).upload_from_file(io.BytesIO(data))
    return file_name
