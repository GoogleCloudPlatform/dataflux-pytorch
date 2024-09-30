import io
import os
import socket
import time
import torch
import torch.multiprocessing as mp
from contextlib import contextmanager
from typing import Generator
from lightning import Trainer
from lightning.pytorch.callbacks import ModelCheckpoint
from lightning.pytorch.demos import (LightningTransformer, Transformer,
                                     WikiText2)
from torch.utils.data import DataLoader, Dataset
from pathlib import Path
from google.cloud import storage

from dataflux_pytorch.lightning.gcs_filesystem import GCSDistributedWriter, GCSDistributedReader
from dataflux_pytorch.lightning.path_utils import parse_gcs_path
from dataflux_core import user_agent
from dataflux_pytorch.lightning import DatafluxLightningCheckpoint
from lightning.pytorch.trainer.states import TrainerFn
from lightning.pytorch.strategies import DDPStrategy
from torch.distributed.fsdp import FullyShardedDataParallel as FSDP
from lightning.pytorch.strategies.fsdp import _METADATA_FILENAME
from torch.distributed.checkpoint import save, load
from torch.nn import Module
from torch.distributed.fsdp.fully_sharded_data_parallel import StateDictType


class DatafluxFSDPStrategy(DDPStrategy):

    def __init__(self, path, project_name, storage_client, model, **kwargs):
        super().__init__(**kwargs)
        self.writer = GCSDistributedWriter(path, project_name, storage_client)
        self.reader = GCSDistributedReader(
            path, project_name, storage_client)
        self.checkpoint_io = DatafluxLightningCheckpoint(
            project_name, storage_client)
        self.model = model
        self.storage_client = storage.Client(project=project_name)
        user_agent.add_dataflux_user_agent(self.storage_client)

    def save_checkpoint(self,
                        checkpoint,
                        filepath,
                        storage_options=None) -> None:
        if storage_options is not None:
            raise TypeError(
                "`FSDPStrategy.save_checkpoint(..., storage_options=...)` is not supported because"
                " `FSDPStrategy` does not use the `CheckpointIO`.")

        path = Path(self.broadcast(filepath))

        converted_state = {"model": checkpoint.pop("state_dict")}
        converted_state.update({
            f"optimizer_{idx}": optim_state
            for idx, optim_state in enumerate(
                checkpoint.pop("optimizer_states", []))
        })
        save(converted_state, checkpoint_id=path, storage_writer=self.writer)

        if self.global_rank == 0:
            self.checkpoint_io.save_checkpoint(checkpoint,
                                               path / _METADATA_FILENAME)

    def get_sharded_state_dict_context(self, module: Module) -> Generator[None, None, None]:

        from torch.distributed.fsdp.api import ShardedOptimStateDictConfig, ShardedStateDictConfig, StateDictType

        state_dict_config = ShardedStateDictConfig(offload_to_cpu=True)
        optim_state_dict_config = ShardedOptimStateDictConfig(
            offload_to_cpu=True)
        state_dict_type_context = FSDP.state_dict_type(
            module=module,
            state_dict_type=StateDictType.SHARDED_STATE_DICT,
            state_dict_config=state_dict_config,
            optim_state_dict_config=optim_state_dict_config,
        )
        return state_dict_type_context  # type: ignore[return-value]

    def load_checkpoint(self, checkpoint_path):
        # broadcast the path from rank 0 to ensure all the states are loaded from a common path
        path = Path(self.broadcast(checkpoint_path))
        print(f" ## PATH in load_checkpoint ## {path}")
        assert self.model is not None
        # assert self.lightning_module is not None

        from torch.distributed.checkpoint.optimizer import load_sharded_optimizer_state_dict

        state_dict_ctx = self.get_sharded_state_dict_context(self.model)
        print(" ### Got state dict ctx ### ")
        with state_dict_ctx:
            module_state = {"model": self.model.state_dict()}
            load(module_state, self.reader)
            print(" ### Loaded Module ### ")
            self.model.load_state_dict(
                module_state["model"], strict=False)
            print(" ### Loaded module state dict ")
            # if self.lightning_module.trainer.state.fn == TrainerFn.FITTING and self.optimizers:

            for idx, optim in enumerate(self.optimizers):
                optim_key = f"optimizer_{idx}"
                optim_state = load_sharded_optimizer_state_dict(
                    model_state_dict=module_state["model"],
                    optimizer_key=optim_key,
                    storage_reader=self.reader,
                )
                print("### Done with optim_state ###")
                flattened_osd = FSDP.optim_state_dict_to_load(
                    optim_state_dict=optim_state[optim_key],
                    model=self.model,
                    optim=optim,
                )
                print("### flattened OSD ###")
                optim.load_state_dict(flattened_osd)

        # Load metadata (anything not a module or optimizer)
        new_path = path / _METADATA_FILENAME
        print(f"### METADTA path = {new_path}")
        metadata = None
        with self.reader.fs.create_stream(path=new_path, mode='rb') as metadata_file:
            print(" ### torch.load(metadata ###)")
            metadata = torch.load(metadata_file)
        return metadata


class DemoTransformer(LightningTransformer):
    def __init__(self, vocab_size: int = 33278, nlayers: int = 2) -> None:
        super().__init__()
        self.model = Transformer(vocab_size=vocab_size, nlayers=nlayers)


def benchmark_strategy(rank, world_size, ckpt_dir_path, project):
    """Benchmark save and load checkpoint."""
    os.environ['WORLD_SIZE'] = str(world_size)
    os.environ['RANK'] = str(rank)
    os.environ['NODE_RANK'] = str(rank)

    # Initialize dummy model and data
    dataset = WikiText2()
    dataloader = DataLoader(dataset, num_workers=1)
    print("### Loaded Data ###")
    model = DemoTransformer(vocab_size=dataset.vocab_size,
                            nlayers=5)

    dataflux_strategy = DatafluxFSDPStrategy(
        path=ckpt_dir_path,
        project_name=project,
        storage_client=None,
        model=model,
    )

    # Create a dummy checkpoint data
    checkpoint_data = {
        "state_dict": model.state_dict(),
        "optimizer_states": []
    }

    accelerator = os.environ.get("ACCELERATOR", "cpu")
    min_epochs_save = os.environ.get("MIN_EPOCHS_SAVE", 4)
    max_epochs_save = os.environ.get("MAX_EPOCHS_SAVE", 5)
    max_steps_save = os.environ.get("MAX_STEPS_SAVE", 3)
    checkpoint_callback = ModelCheckpoint(
        save_top_k=-1,
        every_n_train_steps=1,
        filename="checkpoint-{epoch:02d}-{step:02d}",
        enable_version_counter=True,
    )
    trainer = Trainer(
        default_root_dir=ckpt_dir_path,
        plugins=[],
        callbacks=[checkpoint_callback],
        min_epochs=min_epochs_save,
        max_epochs=max_epochs_save,
        max_steps=max_steps_save,
        accelerator=accelerator,
        strategy=dataflux_strategy,
        num_nodes=world_size,
    )

    # Benchmark save_checkpoint
    start_time = time.time()
    print(f"### Rank {rank}: Starting to save checkpoint")
    trainer.strategy.save_checkpoint(checkpoint_data, os.path.join(
        ckpt_dir_path, f'checkpoints/ckpt_{rank}_0.ckpt/'))
    save_time = time.time() - start_time
    print(f"Rank {rank}: Time taken to SAVE checkpoint: {save_time:.4f} seconds")

    # Benchmark load_checkpoint
    start_time = time.time()
    print(f"### Rank {rank}: Starting to load checkpoint")
    trainer.strategy.load_checkpoint(os.path.join(
        ckpt_dir_path, f'checkpoints/ckpt_{rank}_0.ckpt/'))
    save_time = time.time() - start_time
    print(f"Rank {rank}: Time taken to LOAD checkpoint: {save_time:.4f} seconds")


if __name__ == "__main__":
    world_size = 1  # Number of simulated nodes
    mp.spawn(benchmark_strategy, args=(world_size, "gs://yashsha-us-east1-d/", "gcs-tess"),
             nprocs=world_size, join=True)
