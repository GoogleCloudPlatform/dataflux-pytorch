import os
import socket
import statistics
import time

import torch
import torch.distributed as dist
from demo.lightning.checkpoint.simulated.multiprocessing_train import (
    BenchmarkStrategy, cleanup, format_size, get_tensor_size_bytes,
    time_checkpoint_operation)


def configure_master_addr():
    """Get coordinator IP Address with retries"""
    coordinator_address = ""
    coordinator_ip_address = ""
    if os.environ.get("COORDINATOR_ADDRESS") is not None:
        coordinator_address = os.environ.get("COORDINATOR_ADDRESS")
        coordinator_found = False
        lookup_attempt = 1
        max_coordinator_lookups = 50
        while not coordinator_found and lookup_attempt <= max_coordinator_lookups:
            try:
                coordinator_ip_address = socket.gethostbyname(
                    coordinator_address)
                coordinator_found = True
            except socket.gaierror:
                print(
                    f"Failed to recognize coordinator address {coordinator_address} on"
                    f" attempt {lookup_attempt}, retrying...")
                lookup_attempt += 1
                time.sleep(5)
    print(f"Coordinator IP address: {coordinator_ip_address}")
    os.environ["MASTER_ADDR"] = str(coordinator_ip_address)


def init_processes() -> int:
    """Initializes the distributed environment."""
    world_size = int(os.environ["WORLD_SIZE"])
    job_index = int(os.environ.get("JOB_INDEX", 0))
    job_completion_index = int(os.environ.get("JOB_COMPLETION_INDEX", 0))
    processes_in_job = int(os.environ.get("PROCESSES_IN_JOB", 1))
    rank = job_index * processes_in_job + job_completion_index
    os.environ["NODE_RANK"] = str(rank)

    configure_master_addr()
    torch.distributed.init_process_group(backend='gloo',
                                         rank=rank,
                                         world_size=world_size)
    return rank


def main(world_size: int, layer_size: int, project: str, filepath: str,
         padding_size: int, sample_count: int) -> None:
    rank = init_processes() if os.environ.get("COORDINATOR_ADDRESS") else 0

    benchmark_strategy = BenchmarkStrategy(
        project=project,
        path=filepath,
    )

    state_dict = dict()
    for i in range(padding_size):
        if i % world_size == rank:
            state_dict[f'dummy_tensor_{i}'] = torch.randn(layer_size, 1000)
    # Wait until the state_dict is populated properly accross all the nodes.
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
                {statistics.mean(save_checkpoint_times):.4f} seconds")
        print(f"Time taken to load checkpoint:\
                 {statistics.mean(load_checkpoint_times):.4f} seconds")

        tensor_size_per_instance = 1000 * layer_size * state_dict[
            f'dummy_tensor_0'].element_size()
        tensors_per_rank = padding_size // world_size
        total_size_bytes = total_tensor_count * tensor_size_per_instance * world_size
        print(f"Size of distributed tensors (rank {rank}):\
                 {format_size(total_tensor_count * tensor_size_per_instance)}")
        print(f"Total size of all tensors:\
                 {format_size(total_size_bytes)}")
        print("######################")

    cleanup()


if __name__ == "__main__":
    world_size = int(os.getenv("WORLD_SIZE"))
    layer_size = int(os.getenv("LAYER_SIZE"))
    layer_size = int(os.getenv("LAYER_SIZE"))
    project = os.getenv("PROJECT")
    ckpt_dir_path = os.getenv("CKPT_DIR_PATH")
    sample_count = int(os.getenv("SAMPLE_COUNT", 3))
    padding_size = int(os.getenv("PADDING_SIZE", 4000))
    main(world_size, layer_size, project, ckpt_dir_path, padding_size,
         sample_count)
