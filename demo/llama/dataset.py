# Copyright Lightning AI. Licensed under the Apache License 2.0, see LICENSE file.

# Borrowed from https://github.com/Lightning-AI/lit-llama/blob/main/lit_llama/packed_dataset.py
# with minor modificaitons to use GCS Connector for Pytorch for data loading.

import io
import struct

import numpy as np
from dataflux_core.download import download_single
from google.cloud import storage
from lit_llama.packed_dataset import PackedDataset, PackedDatasetIterator
from torch.utils.data import get_worker_info

dtypes = {
    1: np.uint8,
    2: np.int8,
    3: np.int16,
    4: np.int32,
    5: np.int64,
    6: np.float32,
    7: np.float64,
    8: np.uint16,
}

bucket_name = "<YOUR-BUCKET>"


def code(dtype):
    for k in dtypes.keys():
        if dtypes[k] == dtype:
            return k
    raise ValueError(dtype)


HDR_MAGIC = b"LITPKDS"
HDR_SIZE = 24  # bytes


class DatafluxPackedDataset(PackedDataset):

    def __init__(self,
                 filenames,
                 bucket_name,
                 n_chunks,
                 block_size,
                 seed=12345,
                 shuffle=True,
                 wrap=False,
                 num_processes=1,
                 process_rank=0):
        super().__init(filenames, n_chunks, block_size, seed, shuffle, wrap,
                       num_processes, process_rank)
        self.bucket_name = bucket_name

    def __iter__(self):
        worker_info = get_worker_info()
        num_workers = worker_info.num_workers if worker_info is not None else 1
        worker_id = worker_info.id if worker_info is not None else 0
        num_shards = num_workers * self._num_processes
        shard_id = self._process_rank * num_workers + worker_id

        max_num_files = len(self._filenames) // num_shards * num_shards
        filenames = self._filenames[shard_id:max_num_files:num_shards]

        return DatafluxPackedDatasetIterator(filenames=filenames,
                                             n_chunks=self._n_chunks,
                                             block_size=self._block_size,
                                             seed=self._seed,
                                             shuffle=self._shuffle,
                                             wrap=self._wrap,
                                             bucket_name=self.bucket_name)


class DatafluxPackedDatasetIterator(PackedDatasetIterator):

    def __init__(self, filenames, n_chunks, block_size, seed, shuffle, wrap,
                 bucket_name):
        super().__init__(filenames, n_chunks, block_size, seed, shuffle, wrap)
        self.storage_client = storage.Client()
        self.bucket_name = bucket_name
        self._load_n_chunks()

    def _read(self, path):
        bytes_content = download_single(self.storage_client, self.bucket_name,
                                        path)
        bytes_io = io.BytesIO(bytes_content)
        magic = bytes_io.read(len(HDR_MAGIC))
        assert magic == HDR_MAGIC, "File doesn't match expected format."
        version = struct.unpack("<Q", bytes_io.read(8))
        assert (1, ) == version
        (dtype_code, ) = struct.unpack("<B", bytes_io.read(1))
        dtype = dtypes[dtype_code]
        (chunk_size, ) = struct.unpack("<Q", bytes_io.read(8))
        return dtype, chunk_size, bytes_io

    def _load_n_chunks(self):
        self._close_mmaps()
        self._mmaps = []
        self._buffers = []

        if self._n_chunks > len(self._filenames[self._file_idx:]):
            if not self._wrap:
                raise StopIteration
            else:
                self._file_idx = 0

        for i in range(self._n_chunks):
            filename = self._filenames[self._file_idx + i]
            if self._dtype is None:
                self._dtype, self._chunk_size, bytes_io = self._read(filename)
                self._n_blocks = self._chunk_size // self._block_size

            bytes_io.seek(HDR_SIZE)
            self._buffers.append(bytes_io.read())

        self._file_idx += self._n_chunks
        n_all_blocks = self._n_chunks * self._n_blocks

        self._block_idxs = (self._rng.permutation(n_all_blocks)
                            if self._shuffle else range(n_all_blocks))

        self._curr_idx = 0
