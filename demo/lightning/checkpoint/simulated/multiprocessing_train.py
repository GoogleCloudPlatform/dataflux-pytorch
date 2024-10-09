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
import statistics
import torch
import torch.nn as nn
import torch.distributed as dist
import torch.multiprocessing as mp
import torch.distributed.checkpoint as dist_cp
import os
import time
from typing import Optional, Dict
from lightning.pytorch.strategies import FSDPStrategy
from dataflux_pytorch.lightning.gcs_filesystem import GCSDistributedWriter, GCSDistributedReader

MASTER_ADDR = 'localhost'
MASTER_PORT = '12355'


def write_model_structure(file, model, indent=''):
    for name, module in model.named_children():
        file.write(f"{indent}{name}:\n")
        if list(module.children()):
            write_model_structure(file, module, indent + '  ')
        else:
            file.write(f"{indent}  {module}\n")
            for param_name, param in module.named_parameters():
                file.write(f"{indent}    {param_name}: {param.shape}\n")
                file.write(f"{indent}      Values: {param.data}\n")


def write_full_model(model, filename):
    with open(filename, 'w') as f:
        f.write("Model Structure:\n")
        write_model_structure(f, model)
        f.write("\nModel State Dict:\n")
        for key, value in model.state_dict().items():
            f.write(f"{key}:\n")
            f.write(f"  Shape: {value.shape}\n")
            f.write(f"  Values: {value}\n")

        if hasattr(model, 'dummy_tensors'):
            f.write("\nDummy Tensors:\n")
            for i, tensor in enumerate(model.dummy_tensors):
                f.write(f"dummy_tensor_{i}:\n")
                f.write(f"  Shape: {tensor.shape}\n")
                f.write(f"  Values: {tensor}\n")


def write_state_dict(state_dict, filename):
    with open(filename, 'w') as f:
        f.write("State Dict:\n")
        for key, value in state_dict.items():
            f.write(f"{key}:\n")
            f.write(f"  Shape: {value.shape}\n")
            f.write(f"  Values: {value}\n")


def format_size(size_bytes: int) -> str:
    """Formats bytes into a human-readable size string."""
    if size_bytes < 1024:
        return f"{size_bytes} B"
    elif size_bytes < 1024 * 1024:
        size_kb = size_bytes / 1024
        return f"{size_kb:.2f} KB"
    elif size_bytes < 1024 * 1024 * 1024:
        size_mb = size_bytes / (1024 * 1024)
        return f"{size_mb:.2f} MB"
    else:
        size_gb = size_bytes / (1024 * 1024 * 1024)
        return f"{size_gb:.2f} GB"


def get_tensor_size_bytes(tensor: torch.Tensor) -> int:
    """Calculates the size of a tensor in bytes."""
    return tensor.element_size() * tensor.numel()


def parse_args() -> argparse.Namespace:
    """Parses command line arguments."""
    parser = argparse.ArgumentParser(
        description="Benchmark distributed checkpointing.")
    parser.add_argument("--project", type=str,
                        required=True, help="GCS project ID.")
    parser.add_argument("--ckpt-dir-path", type=str, required=True,
                        help="Path to GCS bucket for checkpoints.")
    parser.add_argument("--layer-size", type=int,
                        default=100, help="Size of the each layer.")
    parser.add_argument("--clear-kernel-cache", action="store_true",
                        default=False, help="Clear kernel cache.")
    parser.add_argument("--sample-count", type=int, default=3,
                        help="Number of samples for benchmarking.")
    parser.add_argument("--padding-size", type=int, default=1000,
                        help="Size of dummy tensors for padding, to control the size of the checkpoint.\
                              Adds approximately 3MB per 100 units of padding size.")
    parser.add_argument("--world-size", type=int, required=True,
                        help="Number of processes in the distributed setup.")
    parser.add_argument(
        "--debug", type=bool, help="In debug mode, the program will print the shape of the tensors present in dictionary as well as write the dictionary to txt file in human readable format.")
    return parser.parse_args()


class BenchmarkStrategy(FSDPStrategy):
    def __init__(self, project: str, path: str, model, world_size: int, **kwargs):

    def __init__(self, project: str, path: str, model, world_size: int, **kwargs):
        super().__init__(**kwargs)
        self.writer = GCSDistributedWriter(path, project, None)
        self.reader = GCSDistributedReader(path, project, None)
        self.model = model
        self.benchmark_world_size = world_size

    def save_checkpoint(self, checkpoint: Dict[str, torch.Tensor], filepath: str, storage_options: Optional[Dict] = None) -> None:
        """
        Saves the model's state dictionary to a specified file path in GCS. torch.distributed.checkpoint.save contains the core logic for saving model shards.
        You can find the source code for FSDP.save_checkpoint
        https://github.com/Lightning-AI/pytorch-lightning/blob/master/src/lightning/fabric/strategies/fsdp.py#L492.
        Args:
            checkpoint (Dict[str, torch.Tensor]): The model's state dictionary containing tensor weights.
            filepath (str): The path where the checkpoint will be saved.
            storage_options (Optional[Dict]): Additional storage options (if any).

        This method uses the GCS writer to save the checkpoint. It is essential for
            maintaining the model's state across training sessions or for recovery after failure.
        """
        dist_cp.save(state_dict=checkpoint, storage_writer=self.writer)

    def load_checkpoint(self, checkpoint_path: str) -> None:
        """
        Loads a model's state dictionary from a specified checkpoint file in GCS. torch.distributed.checkpoint.load contains the core logic of loading sharded model weights.
        You can find the source code for FSDP.load_checkpoint
        https://github.com/Lightning-AI/pytorch-lightning/blob/master/src/lightning/fabric/strategies/fsdp.py#L519.

        For torch.distributed.checkpoint.load to work properly the template_state_dict should have model format. 
        The values of the keys will be overwritten with new values from sharded checkpoint after load is successful.

        Args:
            checkpoint_path (str): The path to the checkpoint file to be loaded.

        This method reads the checkpoint from GCS and updates the model's state dictionary.
        It is crucial for restoring a model's state for continued training or inference.
        Ensure that the model architecture matches the saved state dictionary.
        """
        template_state_dict = self.model.state_dict()
        for key, tensor in template_state_dict.items():
            template_state_dict[key] = torch.empty_like(tensor)
        if hasattr(self.model, 'dummy_tensors'):
            for i, tensor in enumerate(self.model.dummy_tensors):
                template_state_dict[f'dummy_tensor_{i}'] = torch.empty_like(
                    tensor)
        dist_cp.load(state_dict=template_state_dict,
                     storage_reader=self.reader)
        return template_state_dict


class SimpleModel(nn.Module):
    def __init__(self, size):
        super(SimpleModel, self).__init__()
        self.fc1 = nn.Linear(size, size)
        self.fc2 = nn.Linear(size, size)
        self.dummy_tensors = [torch.randn(size, size) for _ in range(500)]

    def forward(self, x):
        return self.fc2(torch.relu(self.fc1(x)))


def setup(rank: int, world_size: int) -> None:
    """Sets up the distributed environment."""
    os.environ['MASTER_ADDR'] = MASTER_ADDR
    os.environ['MASTER_PORT'] = MASTER_PORT
    dist.init_process_group("gloo", rank=rank, world_size=world_size)


def cleanup():
    dist.destroy_process_group()


def time_checkpoint_operation(benchmark_strategy: BenchmarkStrategy, distributed_state_dict: Dict[str, torch.Tensor], filepath: str, sample_count: int, operation: str) -> list:
    """
    Times the save or load operations for checkpoints.

    Args:
        benchmark_strategy (BenchmarkStrategy): The strategy for managing checkpoint operations.
        distributed_state_dict (Dict[str, torch.Tensor]): The model's state dictionary split across processes.
        filepath (str): The path to store/load checkpoints.
        sample_count (int): The number of samples to benchmark.
        operation (str): The operation to perform ('save' or 'load').

    Returns:
        list: A list of times taken for each operation in seconds.

    This function facilitates performance evaluation of checkpoint saving/loading 
    under distributed settings.
    """
    times = []
    for i in range(sample_count):
        checkpoint_path = os.path.join(
            filepath, f'checkpoints/ckpt_{i}.ckpt')
        dist.barrier()
        start_time = time.time()
        if operation == 'save':
            benchmark_strategy.save_checkpoint(
                distributed_state_dict, filepath=checkpoint_path)
        elif operation == 'load':
            benchmark_strategy.load_checkpoint(checkpoint_path=checkpoint_path)
        end_time = time.time()
        times.append(end_time - start_time)
        dist.barrier()  # Synchronize processes
        print(f"Completed iteration {i} for {operation} ...")
    return times


def run_benchmark(rank, world_size: int, layer_size: int, project: str, filepath: str, padding_size: int, sample_count: int, debug: bool):
    setup(rank, world_size)

    model = SimpleModel(layer_size)

    if rank == 0:
        print("Writing initial model structure and parameters to file...")
        write_full_model(model, "initial_model_state.txt")

    if rank == 0 and debug:
        print("Writing initial model structure and parameters to file...")
        write_full_model(model, "initial_model_state.txt")

    benchmark_strategy = BenchmarkStrategy(
        project=project, path=filepath, model=model, world_size=world_size)

    state_dict = model.state_dict()
    for i, tensor in enumerate(model.dummy_tensors):
        state_dict[f'dummy_tensor_{i}'] = tensor

    if rank == 0 and debug:
        print("Writing state dict before saving to file...")
        write_state_dict(state_dict, "state_dict_before_save.txt")
        print("Shapes before saving:", {
            k: v.shape for k, v in state_dict.items()})

    dist.barrier()
    save_checkpoint_times = time_checkpoint_operation(
        benchmark_strategy, state_dict, filepath, sample_count, 'save')
    load_checkpoint_times = time_checkpoint_operation(
        benchmark_strategy, state_dict, filepath, sample_count, 'load')
    if rank == 0:
        print(
            f"Time taken to save checkpoint: {end_time - start_time:.4f} seconds")

    start_time = time.time()
    loaded_state_dict = benchmark_strategy.load_checkpoint()
    dist.barrier()
    end_time = time.time()

    if rank == 0:
        print(
            f"Time taken to load checkpoint: {statistics.mean(load_checkpoint_times):.4f} seconds")
        total_distributed_size_bytes = sum(get_tensor_size_bytes(
            tensor) for tensor in state_dict.values())
        print(
            f"Size of distributed tensors (rank {rank}): {format_size(total_distributed_size_bytes / world_size)}")
        print(
            f"Total size of all tensors (rank {rank}): {format_size(total_distributed_size_bytes )}")
        print("######################")

        if debug:
            print("State dict after loading:")
            write_state_dict(state_dict, "state_dict_after_load.txt")
            print("Shapes after loading:", {
                k: v.shape for k, v in state_dict.items()})

    cleanup()


def main():
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
    mp.spawn(run_benchmark, args=(args.world_size, args.layer_size, args.project, args.ckpt_dir_path, args.padding_size, args.sample_count, args.debug),
             nprocs=args.world_size, join=True)


if __name__ == "__main__":
    world_size = 2
    layer_size = 10
    project = "gcs-tess"
    path = "gs://yashsha-us-east1-d/"
    mp.set_start_method('spawn')
    mp.spawn(run_benchmark, args=(world_size, layer_size,
             project, path), nprocs=world_size, join=True)
