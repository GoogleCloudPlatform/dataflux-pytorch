import argparse
import os
import socket
import time
import torch
import statistics

from typing import Generator
from lightning import Trainer
from lightning.pytorch.callbacks import ModelCheckpoint
from lightning.pytorch.demos import (LightningTransformer, Transformer,
                                     WikiText2)
import torch.distributed
from torch.utils.data import DataLoader
from pathlib import Path
from google.cloud import storage

from dataflux_core import user_agent
from dataflux_pytorch.lightning import DatafluxLightningCheckpoint
from dataflux_pytorch.lightning.gcs_filesystem import GCSDistributedWriter, GCSDistributedReader
from lightning.pytorch.trainer.states import TrainerFn
from lightning.pytorch.strategies import FSDPStrategy
from torch.distributed.fsdp import FullyShardedDataParallel as FSDP

from lightning.pytorch.strategies.fsdp import _METADATA_FILENAME
from torch.distributed.checkpoint import save, load
from torch.nn import Module


class DatafluxFSDPStrategy(FSDPStrategy):

    def __init__(self, path, project_name, storage_client, model, enable_multipart, **kwargs):
        super().__init__(**kwargs)
        self.writer = GCSDistributedWriter(path, project_name, storage_client)
        self.reader = GCSDistributedReader(
            path, project_name, storage_client)
        self.checkpoint_io = DatafluxLightningCheckpoint(
            project_name=project_name, storage_client=storage_client, enable_multipart=enable_multipart)
        self.model = model
        self.storage_client = storage.Client(project=project_name)
        user_agent.add_dataflux_user_agent(self.storage_client)
        # self.save_checkpoints_duration = []
        # self.load_checkpoints_duration = 0
        self.device = torch.device(
            "cuda" if torch.cuda.is_available() else "cpu")

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
        # start_time = time.time()
        path = Path(self.broadcast(checkpoint_path))

        assert self.model is not None
        assert self.lightning_module is not None

        from torch.distributed.checkpoint.optimizer import load_sharded_optimizer_state_dict

        state_dict_ctx = self.get_sharded_state_dict_context(self.model)

        with state_dict_ctx:
            module_state = {"model": self.model.state_dict()}
            load(module_state, self.reader)
            self.model.load_state_dict(
                module_state["model"], strict=self.lightning_module.strict_loading)

            if self.lightning_module.trainer.state.fn == TrainerFn.FITTING and self.optimizers:

                for idx, optim in enumerate(self.optimizers):
                    optim_key = f"optimizer_{idx}"
                    optim_state = load_sharded_optimizer_state_dict(
                        model_state_dict=module_state["model"],
                        optimizer_key=optim_key,
                        storage_reader=self.reader,
                    )
                    flattened_osd = FSDP.optim_state_dict_to_load(
                        optim_state_dict=optim_state[optim_key],
                        model=self.model,
                        optim=optim,
                    )
                    optim.load_state_dict(flattened_osd)

        # Load metadata (anything not a module or optimizer)
        new_path = path / _METADATA_FILENAME
        metadata = None
        with self.reader.fs.create_stream(path=new_path, mode='rb') as metadata_file:
            metadata = torch.load(metadata_file)
        return metadata


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--project", type=str)
    parser.add_argument("--ckpt-dir-path", type=str)
    parser.add_argument("--ckpt_restore_path", type=str)
    parser.add_argument("--save-only-latest",
                        action="store_true",
                        default=False)
    parser.add_argument("--no-dataflux-ckpt",
                        action="store_true",
                        default=False)
    parser.add_argument("--layers", type=int, default=100)
    parser.add_argument("--steps", type=int, default=5)
    parser.add_argument("--enable-multipart",
                        action="store_true",
                        default=False)
    parser.add_argument("--min-epochs-save", type=int, default=4)
    parser.add_argument("--max-epochs-save", type=int, default=5)
    parser.add_argument("--max-steps-save", type=int, default=5)
    parser.add_argument("--min-epochs-restore", type=int, default=4)
    parser.add_argument("--max-epochs-restore", type=int, default=5)
    parser.add_argument("--max-steps-restore", type=int, default=5)
    return parser.parse_args()


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


def init_processes():
    """Initializes the distributed environment."""
    # Get the necessary environment variables from the GKE environment
    world_size = int(os.environ["WORLD_SIZE"])

    job_index = int(os.environ.get("JOB_INDEX"))
    job_completion_index = int(os.environ.get("JOB_COMPLETION_INDEX"))
    processes_in_job = int(os.environ.get("PROCESSES_IN_JOB"))
    rank = job_index * processes_in_job + job_completion_index
    os.environ["NODE_RANK"] = str(rank)

    configure_master_addr()


def main():
    args = parse_args()
    if args.steps < 1:
        raise ValueError("Steps need to greater than 0.")
    torch.cuda.empty_cache()
    if os.environ.get("COORDINATOR_ADDRESS"):
        init_processes()
    dataset = WikiText2()
    dataloader = DataLoader(dataset, num_workers=1)

    model = DemoTransformer(vocab_size=dataset.vocab_size,
                            nlayers=int(os.environ.get("NUM_LAYERS", 10)))
    # Save once per step, and if `save_only_latest`, replace the last checkpoint each time.
    # Replacing is implemented by saving the new checkpoint, and then deleting the previous one.
    # If `save_only_latest` is False, a new checkpoint is created for each step.
    checkpoint_callback = ModelCheckpoint(
        save_top_k=1 if args.save_only_latest else -1,
        every_n_train_steps=1,
        filename="checkpoint-{epoch:02d}-{step:02d}",
        enable_version_counter=True,
    )
    dataflux_strategy = DatafluxFSDPStrategy(
        path=args.ckpt_dir_path,
        project_name=args.project,
        storage_client=None,
        model=model,
        state_dict_type="sharded",
        enable_multipart=args.enable_multipart,
    )

    trainer = Trainer(default_root_dir=args.ckpt_dir_path,
                      plugins=[],
                      callbacks=[checkpoint_callback],
                      min_epochs=args.min_epochs_save,
                      max_epochs=args.max_epochs_save,
                      max_steps=args.max_steps_save,
                      accelerator="gpu",
                      strategy=dataflux_strategy,
                      devices=4,
                      num_nodes=1,
                      )
    trainer.fit(model, dataloader)
    start = time.time()
    for i in range(args.max_steps_save):
        trainer.save_checkpoint(os.path.join(
            args.ckpt_dir_path, f'checkpoints/ckpt_{i}.ckpt/'))
    end = time.time()
    if torch.distributed.get_rank() == 0:
        print("##################################")
        print("Average time to save one checkpoint: " +
              str((end - start) / args.max_steps_save) + " seconds")
        print("##################################")

    load_checkpoint = []
    for i in range(args.max_steps_restore):
        checkpoint_callback = ModelCheckpoint(
            save_top_k=1 if args.save_only_latest else -1,
            every_n_train_steps=0,
            filename="checkpoint-{epoch:02d}-{step:02d}",
            enable_version_counter=True,
        )
        model = DemoTransformer(vocab_size=dataset.vocab_size,
                                nlayers=int(os.environ.get("NUM_LAYERS", 10)))
        new_path = os.path.join(args.ckpt_restore_path, f'ckpt_{i}.ckpt/')
        dataflux_strategy = DatafluxFSDPStrategy(
            path=new_path,
            project_name=args.project,
            storage_client=None,
            model=model,
            state_dict_type="sharded",
            enable_multipart=args.enable_multipart,
        )
        trainer = Trainer(default_root_dir=args.ckpt_dir_path,
                          plugins=[],
                          callbacks=[checkpoint_callback],
                          min_epochs=args.min_epochs_restore,
                          max_epochs=args.max_epochs_restore,
                          max_steps=args.max_steps_restore,
                          accelerator=args.accelerator,
                          strategy=dataflux_strategy,
                          devices=4,
                          num_nodes=1,
                          )
        trainer.fit(model, dataloader, ckpt_path=new_path)
        start = time.time()
        trainer.strategy.load_checkpoint(new_path)
        end = time.time()
        load_checkpoint.append(end-start)

    if torch.distributed.get_rank() == 0:
        print("##################################")
        print("Average time to load one checkpoint: " +
              str(statistics.mean(load_checkpoint)) + " seconds")
        print("##################################")


class DemoTransformer(LightningTransformer):

    def __init__(
        self,
        vocab_size: int = 33278,
        nlayers: int = 2,
    ) -> None:
        super().__init__()
        self.model = Transformer(vocab_size=vocab_size, nlayers=nlayers)


if __name__ == "__main__":

    main()
