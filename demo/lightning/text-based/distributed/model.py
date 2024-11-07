"""
 Copyright 2024 Google LLC

 Licensed under the Apache License, Version 2.0 (the "License");
 you may not use this file except in compliance with the License.
 You may obtain a copy of the License at

      https://www.apache.org/licenses/LICENSE-2.0

 Unless required by applicable law or agreed to in writing, software
 distributed under the License is distributed on an "AS IS" BASIS,
 WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 See the License for the specific language governing permissions and
 limitations under the License.
 """

import argparse
import datetime
import logging
import torch
import math
import os
import socket
import time

import lightning.pytorch as pl


from dataflux_pytorch import dataflux_iterable_dataset as df_iter
from torch import nn, utils, Tensor
from torch.utils import data
from ..demo_model import TextDemoModel, format_data

# This demo was built around the huggingface fineweb dataset.
# Details of the dataset can be found at https://huggingface.co/datasets/HuggingFaceFW/fineweb/tree/main/sample/10BT
# This model has not been optimized for performance and is only intended as a demonstration.

# Example execution:
# python3 ./model.py --project=my-project --bucket=my-fineweb-data --prefix=fineweb/sample/10BT/00 --num-workers=1 --num-nodes=2 --batch-size=128 --epochs=2 --devices=2 --rank=0 --log-level=INFO --limit-train-batches=10 --local=True


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


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--project", type=str)
    parser.add_argument("--bucket", type=str)
    parser.add_argument("--prefix", type=str)
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--num-workers", type=int, default=1)
    parser.add_argument("--devices", type=int, default=1)
    parser.add_argument("--batch-size", type=int, default=100)
    # For clean demo executions, --limit-train-batches should be set to ensure all
    # workers process the same batch count.
    parser.add_argument("--limit-train-batches", type=int, default=None)
    parser.add_argument("--log-level", type=str, default="ERROR")
    parser.add_argument("--num-nodes", type=int, default=1)
    parser.add_argument("--local", default=False, action="store_true")
    parser.add_argument("--rank", type=int, default=0)
    return parser.parse_args()


# This model is configured as an autoencoder. The input is trained to look identical to the output
# despite "squeezing" the data by removing parameters in the training step.
encoder = nn.Sequential(nn.Linear(512, 256), nn.ReLU(), nn.Linear(256, 128))
decoder = nn.Sequential(nn.Linear(128, 256), nn.ReLU(), nn.Linear(256, 512))

# This is an example setup based on a huggingface 10BT parquet dataset. These values should
# be modified to match the format of the parquet files being tested.
cols = [
    'text', 'id', 'dump', 'url', 'date', 'file_path', 'language',
    'language_score', 'token_count'
]

class ParquetIterableDataset(df_iter.DataFluxIterableDataset):

    def __init__(self, columns, batch_size, rank, world_size, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.columns = columns
        self.batch_size = batch_size
        self.current_file = None
        self.columns = columns
        self.world_size = world_size
        self.rank = rank
        """
        A sublcass of the DatafluxIterableDataset, this dataset allows for the reading and batch
        distribution of a parquet file, where batch-size corresponds to rows of the parquet table.
        """

    def __iter__(self):
        """
        Overriding the __iter__ function allows batch size to be used to sub-divide the contents of
        individual parquet files.
        """
        worker_info = data.get_worker_info()
        files_per_node = math.ceil(len(self.objects) / self.world_size)
        start_point = self.rank * files_per_node
        end_point = start_point + files_per_node
        if worker_info is None:
            print("Single-process data loading detected", flush=True)
            for bytes_content in df_iter.dataflux_core.download.dataflux_download_lazy(
                    project_name=self.project_name,
                    bucket_name=self.bucket_name,
                    objects=self.objects[start_point:end_point],
                    storage_client=self.storage_client,
                    dataflux_download_optimization_params=self.
                    dataflux_download_optimization_params,
                    retry_config=self.config.download_retry_config,
            ):
                table = self.data_format_fn(bytes_content)
                for batch in table.iter_batches(batch_size=self.batch_size,
                                                columns=self.columns):
                    yield from batch.to_pylist()
        else:
            # Multi-process data loading. Split the workload among workers.
            # Ref: https://pytorch.org/docs/stable/data.html#torch.utils.data.IterableDataset.
            # For the purpose of this example, this split is only performed at the file level.
            # This means that (num_nodes * num_workers) should be less than or equal to filecount.
            per_worker = int(
                math.ceil(files_per_node / float(worker_info.num_workers)))
            worker_id = worker_info.id
            start = worker_id * per_worker + start_point
            end = min(start + per_worker, end_point)
            logging.info(
                f"-----> Worker {self.rank} downloading {self.objects[start:end]}\n"
            )
            for bytes_content in df_iter.dataflux_core.download.dataflux_download_lazy(
                    project_name=self.project_name,
                    bucket_name=self.bucket_name,
                    objects=self.objects[start:end],
                    storage_client=self.storage_client,
                    dataflux_download_optimization_params=self.
                    dataflux_download_optimization_params,
                    retry_config=self.config.download_retry_config,
            ):
                table = self.data_format_fn(bytes_content)
                for batch in table.iter_batches(batch_size=self.batch_size,
                                                columns=self.columns):
                    yield from batch.to_pylist()


def main():
    args = parse_args()
    logging.basicConfig(level=args.log_level)
    # If running locally, default to localhost port 1234, and use manually specified rank.
    if args.local:
        logging.info(
            "Executing local configuration. Setting MASTER_ADDR/PORT to localhost:1234"
        )
        os.environ["MASTER_ADDR"] = "localhost"
        os.environ["WORLD_SIZE"] = str(args.num_nodes)
        os.environ["MASTER_PORT"] = "1234"
        os.environ["NODE_RANK"] = str(args.rank)
    else:
        # Set environment variables for distributed workload.
        init_processes()
    # The prefix passed here determines which parquet files are read from our bucket.
    # Listing results must be sorted to guarantee each worker gets a different fileset.
    config = df_iter.Config(prefix=args.prefix, sort_listing_results=True)
    my_model = TextDemoModel(encoder, decoder)
    bsize = args.batch_size

    # Construct the lightning trainer and run the fit with our model and custom dataset.
    # Note that limit_train_batches specifies how much of the dataset to train each epoch.
    trainer = pl.Trainer(accelerator='cpu',
                         devices=args.devices,
                         strategy=pl.strategies.DDPStrategy(
                             timeout=datetime.timedelta(minutes=3)),
                         limit_train_batches=args.limit_train_batches,
                         max_epochs=args.epochs,
                         num_nodes=args.num_nodes)

    # Construct our custom dataflux dataset, providing the batch_size at this time to ensure proper
    # sub-processing of each parquet file. Note that rank must be derived from trainer.global_rank
    # to ensure both world_size and device_count value are accounted for.
    dataset = ParquetIterableDataset(
        columns=cols,
        batch_size=args.batch_size,
        rank=trainer.global_rank,
        world_size=trainer.world_size,
        project_name=args.project,
        bucket_name=args.bucket,
        config=config,
        data_format_fn=format_data,
    )

    # Once our custom dataset is defined, it can be provided like any other dataset as an argument
    # to a pytorch Dataloader.
    data_loader = data.DataLoader(
        dataset=dataset,
        batch_size=bsize,
        num_workers=args.num_workers,
        persistent_workers=True,
        pin_memory=True,
    )

    trainer.fit(model=my_model, train_dataloaders=data_loader)


if __name__ == "__main__":
    main()
