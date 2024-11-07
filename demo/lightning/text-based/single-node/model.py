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
import logging
import math

import lightning.pytorch as pl


from dataflux_pytorch import dataflux_iterable_dataset as df_iter
from torch import nn, utils, Tensor
from torch.utils import data
from torch.nn.functional import pad
from transformers import AutoTokenizer
from checkpoint.multinode.train import configure_master_addr
from demo_model import TextDemoModel, format_data

# This demo was built around the huggingface fineweb dataset.
# Details of the dataset can be found at https://huggingface.co/datasets/HuggingFaceFW/fineweb/tree/main/sample/10BT
# This model has not been optimized for performance and is only intended as a demonstration.

# Example execution:
# python3 ./model.py --project=test-project --bucket=my-fineweb-data --batch-size=9 --prefix="fineweb/sample/10BT"


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--project", type=str)
    parser.add_argument("--bucket", type=str)
    parser.add_argument("--prefix", type=str)
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--num-workers", type=int, default=1)
    parser.add_argument("--batch-size", type=int, default=100)
    parser.add_argument("--limit-train-batches", type=int, default=None)
    parser.add_argument("--log-level", type=str, default="ERROR")
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

    def __init__(self, columns, batch_size, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.columns = columns
        self.batch_size = batch_size
        self.current_file = None
        self.columns = columns
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
        if worker_info is None:
            print("Single-process data loading detected", flush=True)
            for bytes_content in df_iter.dataflux_core.download.dataflux_download_lazy(
                    project_name=self.project_name,
                    bucket_name=self.bucket_name,
                    objects=self.objects,
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
            per_worker = int(
                math.ceil(len(self.objects) / float(worker_info.num_workers)))
            worker_id = worker_info.id
            start = worker_id * per_worker
            end = min(start + per_worker, len(self.objects))
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
    # The prefix passed here determines which parquet files are read from our bucket.
    config = df_iter.Config(prefix=args.prefix)
    my_model = TextDemoModel(encoder, decoder)
    bsize = args.batch_size

    # Construct our custom dataflux dataset, providing the batch_size at this time to ensure proper
    # sub-processing of each parquet file.
    dataset = ParquetIterableDataset(
        columns=cols,
        batch_size=args.batch_size,
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

    # Construct the lightning trainer and run the fit with our model and custom dataset.
    # Note that limit_train_batches specifies how much of the dataset to train each epoch.
    trainer = pl.Trainer(limit_train_batches=args.limit_train_batches,
                         max_epochs=args.epochs)
    trainer.fit(model=my_model, train_dataloaders=data_loader)


if __name__ == "__main__":
    main()
