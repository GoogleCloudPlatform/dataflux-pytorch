import torch
import torch.nn as nn
import torch.distributed as dist
import torch.nn as nn
import torch.distributed as dist
import torch.multiprocessing as mp
import torch.distributed.checkpoint as dist_cp
import os
import time
import torch.distributed.checkpoint as dist_cp
import os
import time
from dataflux_pytorch.lightning.gcs_filesystem import GCSDistributedWriter, GCSDistributedReader


class SimpleModel(nn.Module):
    def __init__(self, size):
        super(SimpleModel, self).__init__()
        self.fc1 = nn.Linear(size, size)
        self.fc2 = nn.Linear(size, size)
        # Add dummy tensors
        self.dummy_tensors = [torch.randn(size, size) for _ in range(6)]

    def forward(self, x):
        return self.fc2(torch.relu(self.fc1(x)))


def setup(rank, world_size):
    os.environ['MASTER_ADDR'] = 'localhost'
    os.environ['MASTER_PORT'] = '12355'
    dist.init_process_group("gloo", rank=rank, world_size=world_size)


def cleanup():
    dist.destroy_process_group()


def split_tensor(tensor, world_size, rank):
    numel = tensor.numel()
    split_size = numel // world_size
    start_idx = rank * split_size
    end_idx = start_idx + split_size if rank != world_size - 1 else numel
    return tensor.view(-1)[start_idx:end_idx]


def run_benchmark(rank, world_size, model_size):
    setup(rank, world_size)

    model = SimpleModel(model_size)

    dummy_input = torch.randn(100, model_size)
    _ = model(dummy_input)

    full_state_dict = model.state_dict()
    # Dummy tensors to increase the size of checkpoints.
    for i, tensor in enumerate(model.dummy_tensors):
        full_state_dict[f'dummy_tensor_{i}'] = tensor

    # Distribute all tensors evenly across processes.
    # (This ensures checkpoints have same size accross all the process).
    distributed_state_dict = {}
    for key, tensor in full_state_dict.items():
        split_tensor_chunk = split_tensor(tensor, world_size, rank)
        distributed_state_dict[f"{key}_shard_{rank}"] = split_tensor_chunk

    dist.barrier()

    start_time = time.time()
    dist_cp.save_state_dict(
        state_dict=distributed_state_dict,
        storage_writer=GCSDistributedWriter(
            "gs://yashsha-us-east1-d/", "gcs-tess", None),
    )
    dist.barrier()  # Ensure all processes have finished saving
    end_time = time.time()

    if rank == 0:
        print(
            f"Time taken to save checkpoint: {end_time - start_time:.4f} seconds")
    # Benchmark loading
    empty_state_dict = {}
    dist_cp.save_state_dict(
        state_dict=distributed_state_dict,
        storage_writer=GCSDistributedWriter(
            "gs://yashsha-us-east1-d/", "gcs-tess", None),
    )
    dist.barrier()  # Ensure all processes have finished saving
    end_time = time.time()

    if rank == 0:
        print(
            f"Time taken to save checkpoint: {end_time - start_time:.4f} seconds")
    # Benchmark loading
    empty_state_dict = {}
    start_time = time.time()
    _ = dist_cp.load_state_dict(empty_state_dict,
                                storage_reader=GCSDistributedReader(
                                    "gs://yashsha-us-east1-d/", "gcs-tess", None),
                                )
    dist.barrier()  # Ensure all processes have finished loading
    end_time = time.time()

    if rank == 0:
        print(
            f"Time taken to load checkpoint: {end_time - start_time:.4f} seconds")

    cleanup()
    _ = dist_cp.load_state_dict(empty_state_dict,
                                storage_reader=GCSDistributedReader(
                                    "gs://yashsha-us-east1-d/", "gcs-tess", None),
                                )
    dist.barrier()  # Ensure all processes have finished loading
    end_time = time.time()

    if rank == 0:
        print(
            f"Time taken to load checkpoint: {end_time - start_time:.4f} seconds")

    cleanup()


if __name__ == "__main__":
    world_size = 10  # Number of CPU cores to use
    model_size = 8000  # Size of the model (adjust as needed)

    mp.set_start_method('spawn')
    mp.spawn(run_benchmark, args=(world_size, model_size),
    world_size=10  # Number of CPU cores to use
    model_size=8000  # Size of the model (adjust as needed)

    mp.set_start_method('spawn')
    mp.spawn(run_benchmark, args=(world_size, model_size),
             nprocs=world_size, join=True)
