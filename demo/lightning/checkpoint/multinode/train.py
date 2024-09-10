import io
import os
import socket
import time
import torch

from contextlib import contextmanager
from typing import Generator
from lightning import Trainer
from lightning.pytorch.callbacks import ModelCheckpoint
from lightning.pytorch.demos import (LightningTransformer, Transformer,
                                     WikiText2)
from torch.utils.data import DataLoader
from pathlib import Path
from google.cloud import storage

from dataflux_pytorch.lightning.gcs_filesystem import GCSDistributedWriter, GCSDistributedReader
from dataflux_pytorch.lightning.path_utils import parse_gcs_path
from dataflux_core import user_agent
from dataflux_pytorch.lightning import DatafluxLightningCheckpoint
from lightning.pytorch.trainer.states import TrainerFn
from lightning.pytorch.strategies import FSDPStrategy
from lightning.pytorch.strategies.fsdp import _METADATA_FILENAME
from torch.distributed.checkpoint import save, load
from torch.nn import Module


class DatafluxFSDPStrategy(FSDPStrategy):

    def __init__(self, path, project_name, storage_client, model, **kwargs):
        super().__init__(**kwargs)
        self.writer = GCSDistributedWriter(path, project_name, storage_client)
        self.reader = GCSDistributedReader(
            path, project_name, storage_client)
        self.checkpoint_io = DatafluxLightningCheckpoint(
            project_name, storage_client)
        self.model = model
        self.storage_client = storage.Client(project=project_name,)
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
        from torch.distributed.fsdp import FullyShardedDataParallel as FSDP
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

        from torch.distributed.fsdp import FullyShardedDataParallel as FSDP

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

                # TODO: replace with newer APIs
                # https://github.com/pytorch/pytorch/issues/119800#issuecomment-1942156271
                # the optimizer states must be loaded separately
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
        # bucket, key = parse_gcs_path(path)
        new_path = path / _METADATA_FILENAME
        # new_path = "gs://" + bucket + "/" + key + "/" + _METADATA_FILENAME
        bucket, key = parse_gcs_path(new_path)
        print("####### NEW_PATH #######")
        print(new_path)
        print("###### KEY #######")
        print(key)
        with something(storage_client=self.storage_client, bucket_name=bucket, key=key) as metadata_file:
            print("### TORCH.LOAD (METADATA FILE)#####")
            metadata = torch.load(metadata_file)
        return metadata


@contextmanager
def something(storage_client, bucket_name, key):
    # storage_client = storage.Client(project=project,)
    # user_agent.add_dataflux_user_agent(storage_client)
    # bucket_name, key = parse_gcs_path(path)
    bucket_client = storage_client.bucket(bucket_name)
    blob = bucket_client.blob(key)
    stream = io.BytesIO()
    blob.download_to_file(stream)
    blob_data = blob.download_as_bytes()
    yield io.BytesIO(blob_data)
    # return stream


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


def main(project: str, ckpt_dir_path: str, save_only_latest: bool):
    if os.environ.get("COORDINATOR_ADDRESS"):
        init_processes()
    dataset = WikiText2()
    dataloader = DataLoader(dataset, num_workers=1)

    model = DemoTransformer(vocab_size=dataset.vocab_size,
                            nlayers=int(os.environ.get("NUM_LAYERS", 2)))
    # Save once per step, and if `save_only_latest`, replace the last checkpoint each time.
    # Replacing is implemented by saving the new checkpoint, and then deleting the previous one.
    # If `save_only_latest` is False, a new checkpoint is created for each step.
    checkpoint_callback = ModelCheckpoint(
        save_top_k=1 if save_only_latest else -1,
        every_n_train_steps=1,
        filename="checkpoint-{epoch:02d}-{step:02d}",
        enable_version_counter=True,
    )
    accelerator = os.environ.get("ACCELERATOR", "gpu")
    trainer = Trainer(default_root_dir=ckpt_dir_path,
                      plugins=[],
                      callbacks=[checkpoint_callback],
                      min_epochs=4,
                      max_epochs=5,
                      max_steps=3,
                      accelerator=accelerator,
                      strategy=DatafluxFSDPStrategy(
                          path=ckpt_dir_path,
                          project_name=project,
                          storage_client=None,
                          model=model,
                          state_dict_type="sharded",
                      ),
                      num_nodes=1
                      )
    trainer.fit(model, dataloader)
    new_path = ckpt_dir_path + \
        "lightning_logs/version_0/checkpoints/checkpoint-epoch=00-step=03.ckpt"
    print("### trainer.fit for checkpoint loading...")
    model = DemoTransformer(vocab_size=dataset.vocab_size,
                            nlayers=int(os.environ.get("NUM_LAYERS", 2)))
    trainer = Trainer(default_root_dir=ckpt_dir_path,
                      plugins=[],
                      callbacks=[],
                      min_epochs=4,
                      max_epochs=5,
                      max_steps=3,
                      accelerator=accelerator,
                      strategy=DatafluxFSDPStrategy(
                          path=new_path,
                          project_name=project,
                          storage_client=None,
                          model=model,
                          state_dict_type="sharded",
                      ),
                      num_nodes=1
                      )
    trainer.fit(model, dataloader, ckpt_path=new_path)


class DemoTransformer(LightningTransformer):

    def __init__(
        self,
        vocab_size: int = 33278,
        nlayers: int = 2,
    ) -> None:
        super().__init__()
        self.model = Transformer(vocab_size=vocab_size, nlayers=nlayers)


if __name__ == "__main__":

    main(
        os.getenv("PROJECT"),
        os.getenv("CKPT_DIR_PATH"),
        os.getenv("SAVE_ONLY_LATEST") == "1",
    )
