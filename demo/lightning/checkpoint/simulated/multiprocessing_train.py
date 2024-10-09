import torch
import torch.nn as nn
import torch.distributed as dist
import torch.multiprocessing as mp
import torch.distributed.checkpoint as dist_cp
import os
import time
from lightning.pytorch.strategies import FSDPStrategy
from dataflux_pytorch.lightning.gcs_filesystem import GCSDistributedWriter, GCSDistributedReader


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


def write_distributed_state_dict(distributed_state_dict, filename):
    with open(filename, 'w') as f:
        f.write("Distributed State Dict:\n")
        for key, value in distributed_state_dict.items():
            f.write(f"{key}:\n")
            f.write(f"  Shape: {value.shape}\n")
            f.write(f"  Values: {value}\n")


class BenchmarkStrategy(FSDPStrategy):
    def __init__(self, project: str, path: str, model, world_size: int, **kwargs):
        super().__init__(**kwargs)
        self.writer = GCSDistributedWriter(path, project, None)
        self.reader = GCSDistributedReader(path, project, None)
        self.model = model
        self.benchmark_world_size = world_size

    def save_checkpoint(self, ckpt_state_dict):
        dist_cp.save(state_dict=ckpt_state_dict, storage_writer=self.writer)

    def load_checkpoint(self):
        print(f"### loading checkpoint")
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


def setup(rank, world_size):
    os.environ['MASTER_ADDR'] = 'localhost'
    os.environ['MASTER_PORT'] = '12355'
    dist.init_process_group("gloo", rank=rank, world_size=world_size)


def cleanup():
    dist.destroy_process_group()


def shard_tensor(tensor, world_size, rank):
    if tensor.dim() == 1:
        chunk_size = (tensor.size(0) + world_size - 1) // world_size
        start = rank * chunk_size
        end = min(start + chunk_size, tensor.size(0))
        return tensor[start:end].clone()
    elif tensor.dim() == 2:
        chunk_size = (tensor.size(0) + world_size - 1) // world_size
        start = rank * chunk_size
        end = min(start + chunk_size, tensor.size(0))
        return tensor[start:end, :].clone()
    else:
        raise ValueError(f"Unsupported tensor dimension: {tensor.dim()}")


def run_benchmark(rank, world_size, layer_size, project, path):
    setup(rank, world_size)

    model = SimpleModel(layer_size)

    if rank == 0:
        print("Writing initial model structure and parameters to file...")
        write_full_model(model, "initial_model_state.txt")

    dummy_input = torch.randn(100, layer_size)
    _ = model(dummy_input)

    full_state_dict = model.state_dict()
    for i, tensor in enumerate(model.dummy_tensors):
        full_state_dict[f'dummy_tensor_{i}'] = tensor
    benchmark_strategy = BenchmarkStrategy(
        project=project, path=path, model=model, world_size=world_size)

    distributed_state_dict = {}
    for key, tensor in full_state_dict.items():
        distributed_state_dict[key] = shard_tensor(tensor, world_size, rank)

    if rank == 0:
        print("Writing distributed state dict to file...")
        write_distributed_state_dict(
            distributed_state_dict, "distributed_state_dict.txt")
        # print("Shapes before saving:", {
        #       k: v.shape for k, v in distributed_state_dict.items()})

    dist.barrier()

    start_time = time.time()
    benchmark_strategy.save_checkpoint(distributed_state_dict)
    dist.barrier()
    end_time = time.time()

    if rank == 0:
        print(
            f"Time taken to save checkpoint: {end_time - start_time:.4f} seconds")

    start_time = time.time()
    loaded_state_dict = benchmark_strategy.load_checkpoint()
    dist.barrier()
    end_time = time.time()

    if rank == 0:
        print(
            f"Time taken to load checkpoint: {end_time - start_time:.4f} seconds")
        # print("Shapes after loading:", {
        #       k: v.shape for k, v in loaded_state_dict.items()})
        print("Writing loaded state dict to file...")
        write_distributed_state_dict(
            loaded_state_dict, "loaded_state_dict.txt")

    cleanup()


if __name__ == "__main__":
    world_size = 2
    layer_size = 10
    project = "gcs-tess"
    path = "gs://yashsha-us-east1-d/"
    mp.set_start_method('spawn')
    mp.spawn(run_benchmark, args=(world_size, layer_size,
             project, path), nprocs=world_size, join=True)
