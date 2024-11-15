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
    parser.add_argument("--clear-kernel-cache",
                        action="store_true",
                        default=False,
                        help="Clear kernel cache.")
    parser.add_argument("--sample-count",
                        type=int,
                        default=8,
                        help="Number of samples for benchmarking.")
    parser.add_argument("--world-size",
                        type=int,
                        required=True,
                        help="Number of processes in the distributed setup.")
    parser.add_argument(
        "--use-fsspec",
        action="store_true",
        default=False,
        help=
        ("Use the gcsfs reader/writer for communication with Google Cloud Storage. "
         "If not specified, the custom GCS reader/writer provided by DataFlux (DF) will be used."
         ))
    parser.add_argument(
        "--model-parameter-size",
        type=str,
        required=True,
        help=
        "Model parameter size to simulate. Valid values include 7b, 13b and 70b (case sensitive)"
    )
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


class ModelConfig:

    def __init__(self, model_layers, intermediate_size, hidden_size,
                 attention_head, kv_heads):
        self.model_layers = model_layers
        self.intermediate_size = intermediate_size
        self.hidden_size = hidden_size
        self.attention_head = attention_head
        self.kv_heads = kv_heads

    def __repr__(self):
        return (f"ModelConfig(layers={self.model_layers}, "
                f"intermediate_size={self.intermediate_size}, "
                f"hidden_size={self.hidden_size}, "
                f"attention_head={self.attention_head}, "
                f"kv_heads={self.kv_heads})")


model_7b = ModelConfig(32, 11008, 4096, 32, 32)
model_13b = ModelConfig(40, 13824, 5120, 40, 40)
model_70b = ModelConfig(80, 28672, 8192, 64, 8)

models = {"7b": model_7b, "13b": model_13b, "70b": model_70b}


def create_llama2_7b_state_dict(world_size: int,
                                rank: int,
                                parameters: str,
                                empty: bool = False):
    """
    Creates a state dictionary matching LLAMA2 7B architecture dimensions.

    Parameters:
    world_size (int): The total number of processes/devices.
    rank (int): The current process/device index.
    parameters (str): The parameter size, must be one of '7b', '13b', or '70b'.
    empty (bool, optional): If True, creates an empty state dictionary. Defaults to False.
    """
    state_dict = {}
    if parameters not in models:
        raise ValueError(
            "Invalid parameter size, only valid values are 7b, 13b and 70b")
    model = models[parameters]

    # Model dimensions
    vocab_size = 32000
    hidden_dim = model.hidden_size
    intermediate_dim = model.intermediate_size
    num_layers = model.model_layers

    # Initial embeddings and output normalization and output layer
    if empty:
        state_dict['tok_embeddings.weight'] = torch.empty(
            vocab_size, hidden_dim, dtype=torch.float32).normal_()
        state_dict['norm.weight'] = torch.empty(hidden_dim,
                                                dtype=torch.float32).normal_()
        state_dict['output.weight'] = torch.empty(
            vocab_size, hidden_dim, dtype=torch.float32).normal_()
    else:
        state_dict['tok_embeddings.weight'] = torch.randn(vocab_size,
                                                          hidden_dim,
                                                          dtype=torch.float32)
        state_dict['norm.weight'] = torch.randn(hidden_dim,
                                                dtype=torch.float32)
        state_dict['output.weight'] = torch.randn(vocab_size,
                                                  hidden_dim,
                                                  dtype=torch.float32)

    # Generate layers
    # According to `create_default_local_load_plan` https://github.com/pytorch/pytorch/blob/v2.3.1/torch/distributed/checkpoint/default_planner.py#L227
    # each key will be read only once from the state_dict, hence assigning different names to different tensor will force the load function to only read
    # tensor shard corresponding to given node.
    for layer in range(num_layers):
        if layer % world_size == rank:
            prefix = f'layers.{layer}.'

            # Attention weights
            if empty:
                state_dict[prefix + 'attention.wq.weight'] = torch.empty(
                    hidden_dim, hidden_dim, dtype=torch.float32).normal_()
                state_dict[prefix + 'attention.wk.weight'] = torch.empty(
                    hidden_dim, hidden_dim, dtype=torch.float32).normal_()
                state_dict[prefix + 'attention.wv.weight'] = torch.empty(
                    hidden_dim, hidden_dim, dtype=torch.float32).normal_()
                state_dict[prefix + 'attention.wo.weight'] = torch.empty(
                    hidden_dim, hidden_dim, dtype=torch.float32).normal_()
            else:
                state_dict[prefix + 'attention.wq.weight'] = torch.randn(
                    hidden_dim, hidden_dim, dtype=torch.float32)
                state_dict[prefix + 'attention.wk.weight'] = torch.randn(
                    hidden_dim, hidden_dim, dtype=torch.float32)
                state_dict[prefix + 'attention.wv.weight'] = torch.randn(
                    hidden_dim, hidden_dim, dtype=torch.float32)
                state_dict[prefix + 'attention.wo.weight'] = torch.randn(
                    hidden_dim, hidden_dim, dtype=torch.float32)

            # Attention normalization
            if empty:
                state_dict[prefix + 'attention_norm.weight'] = torch.empty(
                    hidden_dim, dtype=torch.float32).normal_()
            else:
                state_dict[prefix + 'attention_norm.weight'] = torch.randn(
                    hidden_dim, dtype=torch.float32)

            # Feed forward weights
            if empty:
                state_dict[prefix + 'feed_forward.w1.weight'] = torch.empty(
                    intermediate_dim, hidden_dim,
                    dtype=torch.float32).normal_()
                state_dict[prefix + 'feed_forward.w2.weight'] = torch.empty(
                    hidden_dim, intermediate_dim,
                    dtype=torch.float32).normal_()
                state_dict[prefix + 'feed_forward.w3.weight'] = torch.empty(
                    intermediate_dim, hidden_dim,
                    dtype=torch.float32).normal_()
            else:
                state_dict[prefix + 'feed_forward.w1.weight'] = torch.randn(
                    intermediate_dim, hidden_dim, dtype=torch.float32)
                state_dict[prefix + 'feed_forward.w2.weight'] = torch.randn(
                    hidden_dim, intermediate_dim, dtype=torch.float32)
                state_dict[prefix + 'feed_forward.w3.weight'] = torch.randn(
                    intermediate_dim, hidden_dim, dtype=torch.float32)

            # FFN normalization
            if empty:
                state_dict[prefix + 'ffn_norm.weight'] = torch.empty(
                    hidden_dim, dtype=torch.float32).normal_()
            else:
                state_dict[prefix + 'ffn_norm.weight'] = torch.randn(
                    hidden_dim, dtype=torch.float32)

    return state_dict


def time_checkpoint_operation(benchmark_strategy: BenchmarkStrategy,
                              distributed_state_dict: Dict[str, torch.Tensor],
                              filepath: str, sample_count: int, operation: str,
                              rank: int, world_size: int,
                              model_parameter_size: str) -> list:
    times = []
    template_state_dict = create_llama2_7b_state_dict(
        world_size=world_size,
        rank=rank,
        parameters=model_parameter_size,
        empty=True)

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


def run_benchmark(rank, world_size: int, project: str, filepath: str,
                  sample_count: int, use_fsspec: bool,
                  model_parameter_size: str) -> None:
    setup(rank, world_size)

    benchmark_strategy = BenchmarkStrategy(project=project,
                                           path=filepath,
                                           use_fsspec=use_fsspec)
    print(f'Constructing state dict for LLAMA2 {model_parameter_size}')
    state_dict = create_llama2_7b_state_dict(world_size=world_size,
                                             rank=rank,
                                             parameters=model_parameter_size,
                                             empty=False)

    dist.barrier()
    save_checkpoint_times = time_checkpoint_operation(benchmark_strategy,
                                                      state_dict, filepath,
                                                      sample_count, 'save',
                                                      rank, world_size,
                                                      model_parameter_size)

    load_checkpoint_times = time_checkpoint_operation(benchmark_strategy,
                                                      state_dict, filepath,
                                                      sample_count, 'load',
                                                      rank, world_size,
                                                      model_parameter_size)

    if rank == 0:
        print(f"Time taken to save checkpoint:\
                {statistics.mean(save_checkpoint_times):.4f} seconds (stdev {statistics.stdev(save_checkpoint_times):.4f})"
              )
        print(f"All save times: {save_checkpoint_times}")
        print(f"Time taken to load checkpoint:\
                 {statistics.mean(load_checkpoint_times):.4f} seconds (stdev {statistics.stdev(load_checkpoint_times):.4f})"
              )
        print(f"All load times: {load_checkpoint_times}")

        tensor_size_per_instance = sum(v.element_size() * v.numel()
                                       for v in state_dict.values())
        total_size_bytes = tensor_size_per_instance * world_size

        print(f"Size of distributed tensors (rank {rank}):\
                 {format_size(tensor_size_per_instance)}")
        print(f"Total size of all tensors:\
                 {format_size(total_size_bytes)}")

        print("######################")

    cleanup()


def main() -> None:
    """
    Typical usage example:

      python3 -u demo/lightning/checkpoint/simulated/multiprocessing_train.py \
      --project=<gcs_project> \
      --ckpt-dir-path=<path_to_gcs_bucket> \
      --world-size=4 \
      --model-parameter-size=7b
    """
    args = parse_args()

    mp.set_start_method('spawn')
    mp.spawn(run_benchmark,
             args=(args.world_size, args.project, args.ckpt_dir_path,
                   args.sample_count, args.use_fsspec,
                   args.model_parameter_size),
             nprocs=args.world_size,
             join=True)


if __name__ == "__main__":
    main()
