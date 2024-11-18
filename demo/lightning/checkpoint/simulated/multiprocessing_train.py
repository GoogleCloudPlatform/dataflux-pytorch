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
import argparse
import os
import statistics
import time
from typing import Dict, Optional, TextIO

import torch
import torch.distributed as dist
import torch.distributed.checkpoint as dist_cp
import torch.multiprocessing as mp
import torch.nn as nn
from lightning.pytorch.strategies import FSDPStrategy
from torch.distributed.checkpoint import _fsspec_filesystem as FF

from dataflux_pytorch.lightning.gcs_filesystem import (GCSDistributedReader,
                                                       GCSDistributedWriter)

# Constants for distributed setup
MASTER_ADDR = 'localhost'
MASTER_PORT = '12355'

BYTES_PER_KB = 1024
BYTES_PER_MB = BYTES_PER_KB * 1024
BYTES_PER_GB = BYTES_PER_MB * 1024


def write_state_dict_to_file(state_dict: Dict[str, torch.Tensor],
                             filename: str) -> None:
    with open(filename, 'w') as f:
        f.write("State Dict:\n")
        for key, value in state_dict.items():
            f.write(f"{key}:\n")
            f.write(f"  Shape: {value.shape}\n")
            f.write(f"  Values: {value}\n")


def format_size(size_bytes: int) -> str:
    """Formats bytes into a human-readable size string."""
    if size_bytes < BYTES_PER_KB:
        return f"{size_bytes} B"
    elif size_bytes < BYTES_PER_MB:
        size_kb = size_bytes / BYTES_PER_KB
        return f"{size_kb:.2f} KB"
    elif size_bytes < BYTES_PER_GB:
        size_mb = size_bytes / BYTES_PER_MB
        return f"{size_mb:.2f} MB"
    else:
        size_gb = size_bytes / BYTES_PER_GB
        return f"{size_gb:.2f} GB"


def get_tensor_size_bytes(tensor: torch.Tensor) -> int:
    """Calculates the size of a tensor in bytes."""
    return tensor.element_size() * tensor.numel()


def parse_args() -> argparse.Namespace:
    """Parses command line arguments."""
    parser = argparse.ArgumentParser(
        description="Benchmark distributed checkpointing.")
    parser.add_argument("--project",
                        type=str,
                        required=True,
                        help="GCS project ID.")
    parser.add_argument("--ckpt-dir-path",
                        type=str,
                        required=True,
                        help="Path to GCS bucket for checkpoints.")
    parser.add_argument("--layer-size",
                        type=int,
                        default=100,
                        help="Size of each layer.")
    parser.add_argument("--clear-kernel-cache",
                        action="store_true",
                        default=False,
                        help="Clear kernel cache.")
    parser.add_argument("--sample-count",
                        type=int,
                        default=3,
                        help="Number of samples for benchmarking.")
    parser.add_argument("--padding-size",
                        type=int,
                        default=1000,
                        help="Size of dummy tensors for padding.")
    parser.add_argument("--world-size",
                        type=int,
                        required=True,
                        help="Number of processes in the distributed setup.")
    parser.add_argument("--debug",
                        action="store_true",
                        default=False,
                        help="Enable debug mode.")
    parser.add_argument(
        "--use-fsspec",
        action="store_true",
        default=False,
        help=
        ("Use the gcsfs reader/writer for communication with Google Cloud Storage. "
         "If not specified, the custom GCS reader/writer provided by DataFlux (DF) will be used."
         ))
    return parser.parse_args()


class BenchmarkStrategy(FSDPStrategy):

    def __init__(self, project: str, path: str, use_fsspec: bool, **kwargs):
        super().__init__(**kwargs)
        if use_fsspec:
            self.reader = FF.FsspecReader(path)
            self.writer = FF.FsspecWriter(path, sync_files=False)
        else:
            self.writer = GCSDistributedWriter(path, project, None)
            self.reader = GCSDistributedReader(path, project, None)

    def save_checkpoint(self,
                        checkpoint: Dict[str, torch.Tensor],
                        filepath: str,
                        storage_options: Optional[Dict] = None) -> None:
        """
        Saves the model's state dictionary to a specified file path in GCS.
        torch.distributed.checkpoint.save contains the core logic for saving
        model shards.
        Source code for FSDP.save_checkpoint can be found at
        https://github.com/Lightning-AI/pytorch-lightning/blob/master/src/lightning/pytorch/strategies/fsdp.py#L553 .
        Args:
            checkpoint (Dict[str, torch.Tensor]): The model's state dictionary
            containing tensor weights.
            filepath (str): The path where the checkpoint will be saved.
            storage_options (Optional[Dict]): Additional storage options
            (if any).

        This method uses the GCS writer to save the checkpoint.
        """
        dist_cp.save(state_dict=checkpoint,
                     checkpoint_id=filepath,
                     storage_writer=self.writer)

    def load_checkpoint(self, checkpoint_path: str,
                        initial_state_dict: Dict) -> None:
        """
        Loads a model's state dictionary from a specified checkpoint file in
        GCS.
        torch.distributed.checkpoint.load contains the core logic of loading
        sharded model weights.
        Source code for FSDP.load_checkpoint can be found at
        https://github.com/Lightning-AI/pytorch-lightning/blob/master/src/lightning/pytorch/strategies/fsdp.py#L589 .

        For torch.distributed.checkpoint.load to work properly the
        template_state_dict should have model format.
        The values of the keys will be overwritten with new values from
        sharded checkpoint after load is successful.

        Args:
            checkpoint_path (str): The path to the checkpoint file to be
            loaded.

        This method reads the checkpoint from GCS and updates the model's
        state dictionary.
        """
        dist_cp.load(state_dict=initial_state_dict,
                     checkpoint_id=checkpoint_path,
                     storage_reader=self.reader)


def setup(rank: int, world_size: int) -> None:
    """Sets up the distributed environment."""
    os.environ['MASTER_ADDR'] = MASTER_ADDR
    os.environ['MASTER_PORT'] = MASTER_PORT
    dist.init_process_group("gloo", rank=rank, world_size=world_size)


def cleanup() -> None:
    dist.destroy_process_group()


def time_checkpoint_operation(benchmark_strategy: BenchmarkStrategy,
                              distributed_state_dict: Dict[str, torch.Tensor],
                              filepath: str, sample_count: int, operation: str,
                              rank: int, world_size: int, tensor_count: int,
                              tensor_size: int) -> list:
    """
    Times the save or load operations for checkpoints.

    Args:
        benchmark_strategy (BenchmarkStrategy): The strategy for managing
        checkpoint operations.
        distributed_state_dict (Dict[str, torch.Tensor]): The model's state
        dictionary split across processes.
        filepath (str): The path to store/load checkpoints.
        sample_count (int): The number of samples to benchmark.
        operation (str): The operation to perform ('save' or 'load').

    Returns:
        list: A list of times taken for each operation in seconds.

    This function facilitates performance evaluation of checkpoint
    saving/loading under distributed settings.
    """
    times = []
    template_state_dict = dict()
    # According to `create_default_local_load_plan` https://github.com/pytorch/pytorch/blob/main/torch/distributed/checkpoint/default_planner.py#L343
    # each key will be read only once from the state_dict, hence assigning different names to different tensor will force the load function to only read
    # tensor shard corresponding to given node.
    for i in range(tensor_count):
        if i % world_size == rank:
            template_state_dict[f'dummy_tensor_{i}'] = torch.empty(
                tensor_size, 1000)
    for i in range(sample_count):
        checkpoint_path = os.path.join(filepath, f'checkpoints/ckpt_{i}.ckpt')
        dist.barrier()
        print(f"Started iteration {i} for {operation} on rank {rank}...")
        start_time = time.time()
        if operation == 'save':
            benchmark_strategy.save_checkpoint(distributed_state_dict,
                                               filepath=checkpoint_path)
        elif operation == 'load':
            benchmark_strategy.load_checkpoint(
                checkpoint_path=checkpoint_path,
                initial_state_dict=template_state_dict)
        dist.barrier()
        end_time = time.time()
        times.append(end_time - start_time)
        print(f"Completed iteration {i} for {operation} on rank {rank}")
    return times


def run_benchmark(rank, world_size: int, layer_size: int, project: str,
                  filepath: str, padding_size: int, sample_count: int,
                  debug: bool, use_fsspec: bool) -> None:
    setup(rank, world_size)

    benchmark_strategy = BenchmarkStrategy(project=project,
                                           path=filepath,
                                           use_fsspec=use_fsspec)

    state_dict = dict()
    for i in range(padding_size):
        if i % world_size == rank:
            state_dict[f'dummy_tensor_{i}'] = torch.randn(layer_size, 1000)

    if rank == 0 and debug:
        print("Writing state dict before saving to file...")
        write_state_dict_to_file(state_dict, "state_dict_before_save.txt")
        print("Shapes before saving:", {
            k: v.shape
            for k, v in state_dict.items()
        })

    dist.barrier()
    save_checkpoint_times = time_checkpoint_operation(benchmark_strategy,
                                                      state_dict, filepath,
                                                      sample_count, 'save',
                                                      rank, world_size,
                                                      padding_size, layer_size)

    load_checkpoint_times = time_checkpoint_operation(benchmark_strategy,
                                                      state_dict, filepath,
                                                      sample_count, 'load',
                                                      rank, world_size,
                                                      padding_size, layer_size)

    if rank == 0:
        print(f"Time taken to save checkpoint:\
                {statistics.mean(save_checkpoint_times):.4f} seconds (stdev {statistics.stdev(save_checkpoint_times):.4f})"
              )
        print(f"All save times: {save_checkpoint_times}")
        print(f"Time taken to load checkpoint:\
                 {statistics.mean(load_checkpoint_times):.4f} seconds (stdev {statistics.stdev(load_checkpoint_times):.4f})"
              )
        print(f"All load times: {load_checkpoint_times}")

        tensor_size_per_instance = 1000 * layer_size * state_dict[
            f'dummy_tensor_0'].element_size()
        tensors_per_rank = padding_size // world_size
        total_size_bytes = tensors_per_rank * tensor_size_per_instance * world_size
        print(f"Size of distributed tensors (rank {rank}):\
                 {format_size(tensors_per_rank * tensor_size_per_instance)}")
        print(f"Total size of all tensors:\
                 {format_size(total_size_bytes)}")
        print("######################")

        if debug:
            print("State dict after loading:")
            write_state_dict_to_file(state_dict, "state_dict_after_load.txt")
            print("Shapes after loading:", {
                k: v.shape
                for k, v in state_dict.items()
            })

    cleanup()


def main() -> None:
    """
    Typical usage example:

      python3 -u demo/lightning/checkpoint/simulated/multiprocessing_train.py \
      --project=<gcs_project_id> \
      --ckpt-dir-path=<path_to_gcs_bucket> \
      --layer-size=300 \
      --world-size=5
    """
    args = parse_args()

    mp.set_start_method('spawn')
    mp.spawn(run_benchmark,
             args=(args.world_size, args.layer_size, args.project,
                   args.ckpt_dir_path, args.padding_size, args.sample_count,
                   args.debug, args.use_fsspec),
             nprocs=args.world_size,
             join=True)


if __name__ == "__main__":
    main()
