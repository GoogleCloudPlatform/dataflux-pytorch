import argparse
import os
import statistics
import time

import torch
import torch.distributed
from lightning import Trainer
from lightning.pytorch.callbacks import ModelCheckpoint
from lightning.pytorch.demos import WikiText2
from lightning.pytorch.strategies import FSDPStrategy
from torch.utils.data import DataLoader

from demo.lightning.checkpoint.multinode.fsspecfsdp import FSSpecFSDPStrategy
from demo.lightning.checkpoint.multinode.train import (DatafluxFSDPStrategy,
                                                       DemoTransformer,
                                                       init_processes)

DF_FSDP_STRATEGY = "dataflux_fsdp"
FSSPEC_FSDP_STRATEGY = "fsspec_fsdp"
FSDP_STRATEGY = "fsdp"


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--strategy',
        choices=[DF_FSDP_STRATEGY, FSSPEC_FSDP_STRATEGY, FSDP_STRATEGY],
        default=DF_FSDP_STRATEGY)
    return parser.parse_args()


def get_strategy(choice, project, ckpt_dir_path):
    strategy = None
    if choice == DF_FSDP_STRATEGY:
        print("Using DatafluxFSDPStrategy")
        strategy = DatafluxFSDPStrategy(
            path=ckpt_dir_path,
            project_name=project,
            storage_client=None,
            state_dict_type="sharded",
            use_orig_params=False,
        )
    elif choice == FSSPEC_FSDP_STRATEGY:
        print("Using FSSpecFSDPStrategy")
        strategy = FSSpecFSDPStrategy(path=ckpt_dir_path,
                                      state_dict_type="sharded",
                                      use_orig_params=False)
    elif choice == FSDP_STRATEGY:
        print("Using FSDPStrategy.")
        strategy = FSDPStrategy(state_dict_type="sharded",
                                use_orig_params=False)
    else:
        raise ValueError("Invalid strategy.")
    return strategy


def main(ckpt_dir_path: str, ckpt_restore_path: str = ""):
    args = parse_args()
    if os.environ.get("COORDINATOR_ADDRESS"):
        init_processes()
    torch.cuda.empty_cache()
    dataset = WikiText2()
    dataloader = DataLoader(dataset, num_workers=1)

    strategy = get_strategy(args.strategy, os.getenv("PROJECT"), ckpt_dir_path)
    num_save_calls = int(os.environ.get("NUM_SAVE_CALLS", 3))
    num_nodes = int(os.environ.get("NUM_NODES", 1))

    trainer = Trainer(
        enable_checkpointing=False,
        default_root_dir=ckpt_dir_path,
        plugins=[],
        min_epochs=1,
        max_epochs=1,
        max_steps=1,
        accelerator="gpu",
        strategy=strategy,
        devices=os.environ.get("NUM_DEVICES", 'auto'),
        num_nodes=num_nodes,
    )
    with trainer.init_module():
        model = DemoTransformer(vocab_size=dataset.vocab_size,
                                nlayers=int(os.environ.get("NUM_LAYERS", 10)))
    trainer.fit(model, dataloader)
    print(f"Saving checkpoint to {ckpt_dir_path} {num_save_calls} times.")
    start = time.time()
    for i in range(num_save_calls):
        trainer.save_checkpoint(
            os.path.join(ckpt_dir_path, f'checkpoints/ckpt_{i}.ckpt/'))
    end = time.time()
    if torch.distributed.get_rank() == 0:
        print(f"Saved checkpoint to {ckpt_dir_path} {num_save_calls} times.")
    avg_save_time = (end - start) / num_save_calls
    num_load_calls = int(os.environ.get("NUM_LOAD_CALLS", 3))
    load_checkpoint_times = []
    for i in range(num_load_calls):
        new_ckpt_dir_path = os.path.join(ckpt_restore_path, f'ckpt_{i}.ckpt/')
        strategy = get_strategy(args.strategy, os.getenv("PROJECT"),
                                new_ckpt_dir_path)
        trainer = Trainer(
            enable_checkpointing=False,
            default_root_dir=ckpt_dir_path,
            plugins=[],
            min_epochs=1,
            max_epochs=1,
            max_steps=1,
            accelerator="gpu",
            strategy=strategy,
            devices=os.environ.get("NUM_DEVICES", 'auto'),
            num_nodes=num_nodes,
        )
        with trainer.init_module(empty_init=True):
            model = DemoTransformer(vocab_size=dataset.vocab_size,
                                    nlayers=int(
                                        os.environ.get("NUM_LAYERS", 10)))
        trainer.fit(model, dataloader, ckpt_path=new_ckpt_dir_path)
        start = time.time()
        trainer.strategy.load_checkpoint(new_ckpt_dir_path)
        end = time.time()

        if torch.distributed.get_rank() == 0:
            print(f"Loaded checkpoint from {new_ckpt_dir_path}.")
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
    start = time.time()
    main(
        os.getenv("CKPT_DIR_PATH"),
        os.getenv("CKPT_RESTORE_PATH"),
    )
    end = time.time()
    print(f"Benchmark took {end - start} seconds.")
