from demo.lightning.checkpoint.multinode.train import DatafluxFSDPStrategy, init_processes, DemoTransformer

import os
import time
import torch
import statistics

from lightning import Trainer
from lightning.pytorch.callbacks import ModelCheckpoint
from lightning.pytorch.demos import (WikiText2)
import torch.distributed
from torch.utils.data import DataLoader
from lightning.pytorch.strategies import FSDPStrategy

# New imports
from lightning.pytorch.plugins import CheckpointIO
import gcsfs
from lightning.pytorch.utilities import rank_zero_only


import os
import torch
import torch.distributed as dist
import torch.distributed.checkpoint as dcp
from lightning.pytorch.callbacks import Callback
import gcsfs
from fsspec.core import url_to_fs


class GCSShardedCheckpoint(Callback):
    def __init__(self, dirpath, gcs_project, every_n_train_steps=1):
        super().__init__()
        self.dirpath = dirpath
        self.every_n_train_steps = every_n_train_steps
        self.fs = gcsfs.GCSFileSystem(project=gcs_project)

    def on_train_batch_end(self, trainer, pl_module, outputs, batch, batch_idx):
        if (trainer.global_step + 1) % self.every_n_train_steps == 0:
            self.save_checkpoint(trainer, pl_module)

    def save_checkpoint(self, trainer, pl_module):
        if not trainer.is_global_zero:
            return

        checkpoint_dir = os.path.join(
            self.dirpath, f"step_{trainer.global_step}")

        state_dict = {
            "model": trainer.strategy.model,
            "optimizer": trainer.optimizers[0],
        }

        storage_writer = dcp.FileSystemWriter(checkpoint_dir, fs=self.fs)
        dcp.save_state_dict(
            state_dict=state_dict,
            storage_writer=storage_writer,
        )

    def load_checkpoint(self, trainer, pl_module, checkpoint_dir):
        state_dict = {
            "model": trainer.strategy.model,
            "optimizer": trainer.optimizers[0],
        }

        storage_reader = dcp.FileSystemReader(checkpoint_dir, fs=self.fs)
        dcp.load_state_dict(
            state_dict=state_dict,
            storage_reader=storage_reader,
        )


def main(project: str,
         ckpt_dir_path: str,
         save_only_latest: bool,
         ckpt_restore_path: str = ""):
    if os.environ.get("COORDINATOR_ADDRESS"):
        init_processes()
    torch.cuda.empty_cache()
    dataset = WikiText2()
    dataloader = DataLoader(dataset, num_workers=1)

    model = DemoTransformer(vocab_size=dataset.vocab_size,
                            nlayers=int(os.environ.get("NUM_LAYERS", 10)))
    # Save once per step, and if `save_only_latest`, replace the last checkpoint each time.
    # Replacing is implemented by saving the new checkpoint, and then deleting the previous one.
    # If `save_only_latest` is False, a new checkpoint is created for each step.
    min_epochs_save = int(os.environ.get("MIN_EPOCHS_SAVE", 4))
    max_epochs_save = int(os.environ.get("MAX_EPOCHS_SAVE", 5))
    max_steps_save = int(os.environ.get("MAX_STEPS_SAVE", 3))
    # num_nodes = int(os.environ.get("WORLD_SIZE", 2))
    sharded_checkpoint = GCSShardedCheckpoint(
        dirpath=ckpt_dir_path,
        gcs_project=project,
        every_n_train_steps=1,
    )

    dataflux_strategy = FSDPStrategy(
        state_dict_type="sharded",
        state_dict_config={"offload_to_cpu": True, "use_dtensor": True},
    )

    trainer = Trainer(
        default_root_dir=ckpt_dir_path,
        callbacks=[sharded_checkpoint],
        min_epochs=min_epochs_save,
        max_epochs=max_epochs_save,
        max_steps=max_steps_save,
        accelerator="gpu",
        strategy=dataflux_strategy,
        devices=4,
        num_nodes=1,
    )
    trainer.fit(model, dataloader)


if __name__ == "__main__":
    main(
        "gcs-tess",
        "gs://yashsha-benchmarks-us-central1/",
        os.getenv("SAVE_ONLY_LATEST") == "1",
        "gs://yashsha-benchmarks-us-central1/",
    )
