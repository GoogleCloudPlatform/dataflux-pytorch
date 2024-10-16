import argparse
import os
import statistics
import time

import torch
import torch.distributed
from lightning import Trainer
from lightning.pytorch.callbacks import ModelCheckpoint
from lightning.pytorch.demos import WikiText2
from torch.utils.data import DataLoader

from demo.lightning.checkpoint.multinode.fsspecfsdp import FSSpecFSDPStrategy
from demo.lightning.checkpoint.multinode.train import (
    AsyncDatafluxFSDPStrategy, DatafluxFSDPStrategy, DemoTransformer,
    init_processes)

DF_FSDP_STRATEGY = "dataflux_fsdp"
ASYNC_DF_STRATEGY = "async_fsdp"
FSSPEC_FSDP_STRATEGY = "fsspec_fsdp"


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--strategy',
        choices=[DF_FSDP_STRATEGY, ASYNC_DF_STRATEGY, FSSPEC_FSDP_STRATEGY],
        default=DF_FSDP_STRATEGY)
    return parser.parse_args()


def main(project: str,
         ckpt_dir_path: str,
         save_only_latest: bool,
         ckpt_restore_path: str = ""):
    args = parse_args()
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
    checkpoint_callback = ModelCheckpoint(
        save_top_k=1 if save_only_latest else -1,
        every_n_train_steps=1,
        filename="checkpoint-{epoch:02d}-{step:02d}",
        enable_version_counter=True,
    )

    # Parse args to determine which strategy to initialize.
    strategy = None
    if args.strategy == DF_FSDP_STRATEGY:
        print("Using DatafluxFSDPStrategy")
        strategy = DatafluxFSDPStrategy(
            path=ckpt_dir_path,
            project_name=project,
            storage_client=None,
            model=model,
            state_dict_type="sharded",
            use_orig_params=False,
        )
    elif args.strategy == ASYNC_DF_STRATEGY:
        print("Using AsyncDatafluxFSDPStrategy")
        strategy = AsyncDatafluxFSDPStrategy(
            path=ckpt_dir_path,
            project_name=project,
            storage_client=None,
            model=model,
            state_dict_type="sharded",
        )
    elif args.strategy == FSSPEC_FSDP_STRATEGY:
        print("Using FSSpecFSDPStrategy")
        strategy = FSSpecFSDPStrategy(path=ckpt_dir_path,
                                      model=model,
                                      state_dict_type="sharded",
                                      use_orig_params=False)
    else:
        raise Exception("Invalid strategy choice")

    min_epochs_save = int(os.environ.get("MIN_EPOCHS_SAVE", 4))
    max_epochs_save = int(os.environ.get("MAX_EPOCHS_SAVE", 5))
    max_steps_save = int(os.environ.get("MAX_STEPS_SAVE", 3))
    num_nodes = int(os.environ.get("NUM_NODES", 1))

    trainer = Trainer(
        default_root_dir=ckpt_dir_path,
        plugins=[],
        callbacks=[checkpoint_callback],
        min_epochs=min_epochs_save,
        max_epochs=max_epochs_save,
        max_steps=max_steps_save,
        accelerator="gpu",
        strategy=strategy,
        devices=os.environ.get("NUM_DEVICES", 'auto'),
        num_nodes=num_nodes,
    )
    trainer.fit(model, dataloader)
    print(f"Saving checkpoint to {ckpt_dir_path} {max_steps_save} times.")
    start = time.time()
    for i in range(max_steps_save):
        trainer.save_checkpoint(
            os.path.join(ckpt_dir_path, f'checkpoints/ckpt_{i}.ckpt/'))
    end = time.time()
    if torch.distributed.get_rank() == 0:
        print(f"Saved checkpoint to {ckpt_dir_path} {max_steps_save} times.")
    avg_save_time = (end - start) / max_steps_save
    min_epochs_restore = int(os.environ.get("MIN_EPOCHS_RESTORE", 4))
    max_epochs_restore = int(os.environ.get("MAX_EPOCHS_RESTORE", 5))
    max_steps_restore = int(os.environ.get("MAX_STEPS_RESTORE", 3))
    load_checkpoint_times = []
    for i in range(max_steps_restore):
        checkpoint_callback = ModelCheckpoint(
            save_top_k=1 if save_only_latest else -1,
            every_n_train_steps=0,
            filename="checkpoint-{epoch:02d}-{step:02d}",
            enable_version_counter=True,
        )
        model = DemoTransformer(vocab_size=dataset.vocab_size,
                                nlayers=int(os.environ.get("NUM_LAYERS", 10)))
        new_path = os.path.join(ckpt_restore_path, f'ckpt_{i}.ckpt/')
        strategy = None
        if args.strategy == DF_FSDP_STRATEGY:
            print("Using DatafluxFSDPStrategy")
            strategy = DatafluxFSDPStrategy(
                path=new_path,
                project_name=project,
                storage_client=None,
                model=model,
                state_dict_type="sharded",
                use_orig_params=False,
            )
        else:
            print("Using FSSpecFSDPStrategy")
            strategy = FSSpecFSDPStrategy(path=new_path,
                                          model=model,
                                          state_dict_type="sharded",
                                          use_orig_params=False)
        trainer = Trainer(
            default_root_dir=ckpt_dir_path,
            plugins=[],
            callbacks=[checkpoint_callback],
            min_epochs=min_epochs_restore,
            max_epochs=max_epochs_restore,
            max_steps=max_steps_restore,
            accelerator="gpu",
            strategy=strategy,
            devices=os.environ.get("NUM_DEVICES", 'auto'),
            num_nodes=num_nodes,
        )
        trainer.fit(model, dataloader, ckpt_path=new_path)
        start = time.time()
        trainer.strategy.load_checkpoint(new_path)
        end = time.time()

        if torch.distributed.get_rank() == 0:
            print(f"Loaded checkpoint from {new_path}.")
        load_checkpoint_times.append(end - start)

    if torch.distributed.get_rank() == 0:
        avg_load_time = statistics.mean(load_checkpoint_times)
        print("##################################")
        print("Average time to save one checkpoint: " + str(avg_save_time) +
              " seconds")
        print("Average time to load one checkpoint: " + str(avg_load_time) +
              " seconds")
        print("##################################")


if __name__ == "__main__":
    main(
        os.getenv("PROJECT"),
        os.getenv("CKPT_DIR_PATH"),
        os.getenv("SAVE_ONLY_LATEST") == "1",
        os.getenv("CKPT_RESTORE_PATH"),
    )
