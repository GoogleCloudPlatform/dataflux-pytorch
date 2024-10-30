import os

import torch
import torch.distributed
from lightning import Trainer
from lightning.pytorch.callbacks import ModelCheckpoint
from lightning.pytorch.demos import WikiText2
from torch.utils.data import DataLoader

from demo.lightning.checkpoint.multinode.strategies import (
    AsyncDatafluxFSDPStrategy, DatafluxFSDPStrategy)
from demo.lightning.checkpoint.multinode.train import (DemoTransformer,
                                                       init_processes)

ASYNC_DF_STRATEGY = "async_dataflux_fsdp"
DF_STRATEGY = "dataflux_fsdp"


def main():
    project = os.getenv("PROJECT")
    num_nodes = int(os.environ.get("NUM_NODES", 1))
    devices = os.environ.get("NUM_DEVICES", 'auto')

    strategy_flag = os.getenv("STRATEGY")
    ckpt_dir_path = os.getenv("CKPT_DIR_PATH")
    num_layers = int(os.environ.get("NUM_LAYERS", 10))
    min_epochs = int(os.environ.get("MIN_EPOCHS", 4))
    max_epochs = int(os.environ.get("MAX_EPOCHS", 5))
    max_steps = int(os.environ.get("MAX_STEPS", 3))
    steps_per_save = int(os.environ.get("STEPS_PER_SAVE", 1))
    run_count = int(os.getenv("RUN_COUNT", 1))

    rank = 0
    if os.environ.get("COORDINATOR_ADDRESS"):
        rank = init_processes()
        torch.distributed.init_process_group("gloo",
                                             rank=rank,
                                             world_size=num_nodes)

    torch.cuda.empty_cache()

    dataset = WikiText2()
    dataloader = DataLoader(dataset, num_workers=1)

    run_total = 0
    for i in range(run_count):

        model = DemoTransformer(vocab_size=dataset.vocab_size,
                                nlayers=num_layers)

        checkpoint_callback = ModelCheckpoint(
            save_top_k=-1,
            every_n_train_steps=steps_per_save,
            filename="checkpoint-{epoch:02d}-{step:02d}",
            enable_version_counter=True,
            dirpath=ckpt_dir_path,
        )

        if strategy_flag == ASYNC_DF_STRATEGY:
            strategy = AsyncDatafluxFSDPStrategy(
                path=ckpt_dir_path,
                project_name=project,
                storage_client=None,
                model=model,
                state_dict_type="sharded",
                use_orig_params=False,
            )
        elif strategy_flag == DF_STRATEGY:
            strategy = DatafluxFSDPStrategy(
                path=ckpt_dir_path,
                project_name=project,
                storage_client=None,
                model=model,
                state_dict_type="sharded",
                use_orig_params=False,
            )
        else:
            raise ValueError("Unexpected value for STRATEGY env var")

        trainer = Trainer(
            default_root_dir=ckpt_dir_path,
            plugins=[],
            callbacks=[checkpoint_callback],
            max_steps=max_steps,
            min_epochs=min_epochs,
            max_epochs=max_epochs,
            accelerator="gpu",
            strategy=strategy,
            devices=devices,
            num_nodes=num_nodes,
        )

        init_start_event = torch.cuda.Event(enable_timing=True)
        init_end_event = torch.cuda.Event(enable_timing=True)

        init_start_event.record()
        trainer.fit(model, dataloader)
        init_end_event.record()

        total_secs = init_start_event.elapsed_time(init_end_event) / 1000
        print(
            f"Individual run {i+1} of {run_count} trainer.fit() #{rank} took {total_secs} seconds."
        )
        run_total += total_secs

    # All runs complete.
    print(f"Average execution run time: {run_total/run_count} seconds.")

    # Cleanup.
    torch.distributed.destroy_process_group()


if __name__ == "__main__":
    main()
