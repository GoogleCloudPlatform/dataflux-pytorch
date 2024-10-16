import os
import socket
import statistics
import time

import torch
import torch.distributed as dist
from demo.lightning.checkpoint.simulated.multiprocessing_train import (
    BenchmarkStrategy, SimpleModel, cleanup, format_size,
    get_tensor_size_bytes, time_checkpoint_operation)


def configure_master_addr():
    """Get coordinator IP Address with retries"""
    coordinator_address = ""
    coordinator_ip_address = ""
    if os.environ.get("COORDINATOR_ADDRESS") is not None:
        coordinator_address = os.environ.get("COORDINATOR_ADDRESS")
        coordinator_found = False
        lookup_attempt = 1
        max_coordinator_lookups = 50
        while (not coordinator_found
               and lookup_attempt <= max_coordinator_lookups):
            try:
                coordinator_ip_address = socket.gethostbyname(
                    coordinator_address)
                coordinator_found = True
            except socket.gaierror:
                print(f"Failed to recognize coordinator address \
                        {coordinator_address} on attempt {lookup_attempt},\
                             retrying...")
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


def main(num_nodes: int, layer_size: int, project: str, filepath: str,
         padding_size: int, sample_size: int) -> None:
    rank = init_processes() if os.environ.get("COORDINATOR_ADDRESS") else 0
    model = SimpleModel(layer_size, padding_size)

    full_state_dict = model.state_dict()
    for i, tensor in enumerate(model.dummy_tensors):
        full_state_dict[f'dummy_tensor_{i}'] = tensor

    benchmark_strategy = BenchmarkStrategy(project=project,
                                           path=filepath,
                                           model=model)

    state_dict = model.state_dict()
    for i, tensor in enumerate(model.dummy_tensors):
        state_dict[f'dummy_tensor_{i}'] = tensor
    dist.barrier()

    save_checkpoint_times = time_checkpoint_operation(benchmark_strategy,
                                                      state_dict, filepath,
                                                      sample_size, 'save',
                                                      model)

    load_checkpoint_times = time_checkpoint_operation(benchmark_strategy,
                                                      state_dict, filepath,
                                                      sample_size, 'load',
                                                      model)

    if rank == 0:
        print(f"Time taken to save checkpoint:\
                {statistics.mean(save_checkpoint_times):.4f} seconds")
        print(f"Time taken to load checkpoint:\
                 {statistics.mean(load_checkpoint_times):.4f} seconds")
        total_distributed_size_bytes = sum(
            get_tensor_size_bytes(tensor) for tensor in state_dict.values())
        print(f"Size of each shard\
                 {format_size(total_distributed_size_bytes / num_nodes)}")
        print(f"Total size of total checkpoint:\
                 {format_size(total_distributed_size_bytes)}")
        print("######################")

    cleanup()


if __name__ == "__main__":
    num_nodes = int(os.environ.get("WORLD_SIZE", 1))
    layer_size = int(os.getenv("LAYER_SIZE"))
    project = os.getenv("PROJECT")
    path = os.getenv("CKPT_DIR_PATH")
    sample_size = int(os.getenv("SAMPLE_COUNT", 3))
    padding_size = int(os.getenv("PADDING_SIZE", 4000))
    main(num_nodes, layer_size, project, path, padding_size, sample_size)
