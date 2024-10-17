import os

import gcsfs
import torch

from pathlib import Path
from typing import Generator
from lightning.pytorch.trainer.states import TrainerFn
from lightning.pytorch.strategies import FSDPStrategy
from torch.distributed.fsdp import FullyShardedDataParallel as FSDP

from lightning.pytorch.strategies.fsdp import _METADATA_FILENAME
from torch.distributed.checkpoint import save, load
from torch.nn import Module
from torch.distributed.checkpoint import _fsspec_filesystem as FF
from dataflux_core import user_agent
from google.cloud import storage
from dataflux_pytorch.lightning import DatafluxLightningCheckpoint
from dataflux_pytorch.lightning.gcs_filesystem import (GCSDistributedReader,
                                                       GCSDistributedWriter)


class DatafluxFSDPStrategy(FSDPStrategy):

    def __init__(self, path, project_name, storage_client, model, **kwargs):
        super().__init__(**kwargs)
        self.writer = GCSDistributedWriter(path, project_name, storage_client)
        self.reader = GCSDistributedReader(path, project_name, storage_client)
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
                "`FSDPStrategy.save_checkpoint(..., storage_options=...)` is\
                not supported because`FSDPStrategy` does not use the \
                    `CheckpointIO`.")

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

    def get_sharded_state_dict_context(
            self, module: Module) -> Generator[None, None, None]:

        from torch.distributed.fsdp.api import (ShardedOptimStateDictConfig,
                                                ShardedStateDictConfig,
                                                StateDictType)

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
        # broadcast the path from rank 0 to ensure all the states are loaded \
        # from a common path.
        path = Path(self.broadcast(checkpoint_path))

        assert self.model is not None
        assert self.lightning_module is not None

        from torch.distributed.checkpoint.optimizer import \
            load_sharded_optimizer_state_dict

        state_dict_ctx = self.get_sharded_state_dict_context(self.model)

        with state_dict_ctx:
            module_state = {"model": self.model.state_dict()}
            load(module_state, self.reader)
            self.model.load_state_dict(
                module_state["model"],
                strict=self.lightning_module.strict_loading)

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
        with self.reader.fs.create_stream(path=new_path,
                                          mode='rb') as metadata_file:
            metadata = torch.load(metadata_file)
        return metadata


class FSSpecFSDPStrategy(FSDPStrategy):

    def __init__(self, path, model, **kwargs):
        super().__init__(**kwargs)
        self.model = model
        self.path = path
        self.reader = FF.FsspecReader(path)
        self.writer = FF.FsspecWriter(self.path, sync_files=False)
        self.bucket = gcsfs.GCSFileSystem()

    def save_checkpoint(self,
                        checkpoint,
                        filepath,
                        storage_options=None) -> None:
        if storage_options is not None:
            raise TypeError(
                "`FSDPStrategy.save_checkpoint(..., storage_options=...)` is not supported because"
                " `FSDPStrategy` does not use the `CheckpointIO`.")

        # broadcast the path from rank 0 to ensure all the states are loaded from a common path
        self.broadcast(filepath)

        converted_state = {"model": checkpoint.pop("state_dict")}
        converted_state.update({
            f"optimizer_{idx}": optim_state
            for idx, optim_state in enumerate(
                checkpoint.pop("optimizer_states", []))
        })
        save(converted_state,
             checkpoint_id=filepath,
             storage_writer=self.writer)

        with self.bucket.open(os.path.join(filepath, _METADATA_FILENAME),
                              'wb') as f:
            torch.save(checkpoint, f)

    def get_sharded_state_dict_context(
            self, module: Module) -> Generator[None, None, None]:

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
        self.broadcast(checkpoint_path)

        assert self.model is not None
        assert self.lightning_module is not None

        from torch.distributed.checkpoint.optimizer import load_sharded_optimizer_state_dict

        state_dict_ctx = self.get_sharded_state_dict_context(self.model)

        with state_dict_ctx:
            module_state = {"model": self.model.state_dict()}
            load(module_state, self.reader)
            self.model.load_state_dict(
                module_state["model"],
                strict=self.lightning_module.strict_loading)

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
        new_path = os.path.join(checkpoint_path, _METADATA_FILENAME)
        metadata = None
        with self.reader.fs.create_stream(path=new_path,
                                          mode='rb') as metadata_file:
            metadata = torch.load(metadata_file)
        return metadata


class CustomFSDPStrategy(FSDPStrategy):

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def save_checkpoint(self,
                        checkpoint,
                        filepath,
                        storage_options=None) -> None:
        super().save_checkpoint(checkpoint, filepath, storage_options)
        if self.global_rank != 0:
            torch.save(checkpoint, os.path.join(filepath, _METADATA_FILENAME))
