import os
import socket
import time

import torch
import torch.optim
from lightning import Trainer
from lightning.pytorch.callbacks import ModelCheckpoint
from lightning.pytorch.demos import (LightningTransformer, Transformer,
                                     WikiText2)
from torch.utils.data import DataLoader

from .strategies import DatafluxFSDPStrategy


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
                print(f"Failed to recognize coordinator address\
                         {coordinator_address} on attempt \
                            {lookup_attempt}, retrying...")
                lookup_attempt += 1
                time.sleep(5)
    print(f"Coordinator IP address: {coordinator_ip_address}")
    os.environ["MASTER_ADDR"] = str(coordinator_ip_address)


def init_processes():
    """Initializes the distributed environment."""
    # Get the necessary environment variables from the GKE environment.
    job_index = int(os.environ.get("JOB_INDEX"))
    job_completion_index = int(os.environ.get("JOB_COMPLETION_INDEX"))
    processes_in_job = int(os.environ.get("PROCESSES_IN_JOB"))
    rank = job_index * processes_in_job + job_completion_index
    os.environ["NODE_RANK"] = str(rank)

    configure_master_addr()


def main(project: str,
         ckpt_dir_path: str,
         save_only_latest: bool,
         ckpt_restore_path: str = ""):
    if os.environ.get("COORDINATOR_ADDRESS"):
        init_processes()
    dataset = WikiText2()
    dataloader = DataLoader(dataset, num_workers=1)

    model = DemoTransformer(vocab_size=dataset.vocab_size,
                            nlayers=int(os.environ.get("NUM_LAYERS", 2)))
    # Save once per step, and if `save_only_latest`, replace the last \
    # checkpoint each time.
    # Replacing is implemented by saving the new checkpoint, and then deleting\
    # the previous one.
    # If `save_only_latest` is False, a new checkpoint is created for each \
    # step.
    checkpoint_callback = ModelCheckpoint(
        save_top_k=1 if save_only_latest else -1,
        every_n_train_steps=1,
        filename="checkpoint-{epoch:02d}-{step:02d}",
        enable_version_counter=True,
    )
    accelerator = os.environ.get("ACCELERATOR", "cpu")
    min_epochs_save = int(os.environ.get("MIN_EPOCHS_SAVE", 4))
    max_epochs_save = int(os.environ.get("MAX_EPOCHS_SAVE", 5))
    max_steps_save = int(os.environ.get("MAX_STEPS_SAVE", 3))
    trainer = Trainer(default_root_dir=ckpt_dir_path,
                      plugins=[],
                      callbacks=[checkpoint_callback],
                      min_epochs=min_epochs_save,
                      max_epochs=max_epochs_save,
                      max_steps=max_steps_save,
                      accelerator=accelerator,
                      strategy=DatafluxFSDPStrategy(
                          project_name=project,
                          storage_client=None,
                          state_dict_type="sharded",
                      ),
                      num_nodes=int(os.environ.get("WORLD_SIZE", 5)))
    trainer.fit(model, dataloader)

    print("Restoring checkpoints ...")
    min_epochs_restore = int(os.environ.get("MIN_EPOCHS_RESTORE", 4))
    max_epochs_restore = int(os.environ.get("MAX_EPOCHS_RESTORE", 5))
    max_steps_restore = int(os.environ.get("MAX_STEPS_RESTORE", 3))
    model = DemoTransformer(vocab_size=dataset.vocab_size,
                            nlayers=int(os.environ.get("NUM_LAYERS", 2)))
    trainer = Trainer(default_root_dir=ckpt_dir_path,
                      plugins=[],
                      callbacks=[],
                      min_epochs=min_epochs_restore,
                      max_epochs=max_epochs_restore,
                      max_steps=max_steps_restore,
                      accelerator=accelerator,
                      strategy=DatafluxFSDPStrategy(
                          project_name=project,
                          storage_client=None,
                          state_dict_type="sharded",
                      ),
                      num_nodes=int(os.environ.get("WORLD_SIZE", 5)))
    trainer.fit(model, dataloader, ckpt_path=ckpt_restore_path)


class DemoTransformer(LightningTransformer):

    def __init__(
        self,
        vocab_size: int = 33278,
        nlayers: int = 2,
        optimizer: str = "sgd",
    ) -> None:
        super().__init__()
        self.optimizer = optimizer
        self.model = Transformer(vocab_size=vocab_size, nlayers=nlayers)

    def configure_optimizers(self) -> torch.optim.Optimizer:
        # Use self.trainer.model.parameters so that we can set
        # use_orig_params=False on the Strategy. Using AdamW also results in a
        # checkpoint size roughly 20% of used GPU memory.
        if self.optimizer == "adamw":
            return torch.optim.AdamW(self.trainer.model.parameters(), lr=0.1)
        else:
            return torch.optim.SGD(self.trainer.model.parameters(), lr=0.1)


if __name__ == "__main__":

    main(
        os.getenv("PROJECT"),
        os.getenv("CKPT_DIR_PATH"),
        os.getenv("SAVE_ONLY_LATEST") == "1",
        os.getenv("CKPT_RESTORE_PATH"),
    )
