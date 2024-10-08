import os
import socket
import statistics
import time

import torch
from demo.lightning.checkpoint.simulated.multiprocessing_train import (
    BenchmarkStrategy,
    SimpleModel,
    cleanup,
    format_size,
    get_tensor_size_bytes,
    split_tensor,
    time_checkpoint_operation,
)


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
    torch.distributed.init_process_group(
        backend='gloo', rank=rank, world_size=world_size)
    return rank


def main(world_size: int, model_size: int, project: str, filepath: str, padding_size: int, sample_size: int) -> None:
    rank = init_processes() if os.environ.get("COORDINATOR_ADDRESS") else 0
    model = SimpleModel(model_size, padding_size)

    dummy_input = torch.randn(100, model_size)
    _ = model(dummy_input)

    full_state_dict = model.state_dict()
    for i, tensor in enumerate(model.dummy_tensors):
        full_state_dict[f'dummy_tensor_{i}'] = tensor

    benchmark_strategy = BenchmarkStrategy(
        project=project, path=filepath, model=model)

    distributed_state_dict = {f"{key}_shard_{rank}": split_tensor(
        tensor, world_size, rank) for key, tensor in full_state_dict.items()}

    save_checkpoint_times = time_checkpoint_operation(
        benchmark_strategy, distributed_state_dict, filepath, sample_size, 'save')
    load_checkpoint_times = time_checkpoint_operation(
        benchmark_strategy, distributed_state_dict, filepath, sample_size, 'load')

    if rank == 0:
        print("######################")
        print(
            f"Time taken to save checkpoint: {statistics.mean(save_checkpoint_times):.4f} seconds")
        print(
            f"Time taken to load checkpoint: {statistics.mean(load_checkpoint_times):.4f} seconds")
        total_distributed_size_bytes = sum(get_tensor_size_bytes(
            tensor) for tensor in distributed_state_dict.values())
        print(
            f"Size of distributed tensors (rank {rank}): {format_size(total_distributed_size_bytes)}")
        print(
            f"Total size of all tensors (rank {rank}): {format_size(total_distributed_size_bytes * world_size)}")
        print("######################")

    cleanup()


if __name__ == "__main__":
    world_size = int(os.getenv("WORLD_SIZE"))
    model_size = int(os.getenv("NUM_LAYERS"))
    project = os.getenv("PROJECT")
    path = os.getenv("CKPT_DIR_PATH")
    sample_size = int(os.getenv("SAMPLE_SIZE", 3))
    padding_size = int(os.getenv("PADDING_SIZE", 4000))
    main(world_size, model_size,
         project, path, padding_size, sample_size)
