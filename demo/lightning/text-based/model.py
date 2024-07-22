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
import torch
import math

import lightning.pytorch as pl
import pyarrow as pa
import pyarrow.parquet as pq

from dataflux_pytorch import dataflux_iterable_dataset as df_iter
from torch import optim, nn, utils, Tensor
from torch.utils import data
from torch.nn.functional import pad
from transformers import AutoTokenizer

# This demo was built around the huggingface fineweb dataset.
# Details of the dataset can be found at https://huggingface.co/datasets/HuggingFaceFW/fineweb/tree/main/sample/10BT

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

# This is a pre-trained tokenizer designed for text-based use cases. It is configured to expect a maximum
# input size of 512. A user can speciy any tokenizer they wish, or design their own as long as the output
# parameter count matches the input of the encoder (in this case 512).
tokenizer = AutoTokenizer.from_pretrained("bert-base-cased")
INPUT_SIZE = 512


class TextDemoModel(pl.LightningModule):
    """
    This model subclasses the base LightningModule and configures itself as an autoencoder.
    """

    def __init__(self, encoder, decoder):
        """
        The encoder and decoder sequences to use are passed directly into this model.
        """
        super().__init__()
        self.encoder = encoder
        self.decoder = decoder

    def configure_optimizers(self):
        """
        This function sets the loss parameter (lr) which determines the rate at which the model "learns".
        The parameters pass is a direct-passthrough of any existing parameters from the LightingModule.
        """
        optimizer = optim.Adam(self.parameters(), lr=1e-3)
        return optimizer

    def training_step(self, train_batch, batch_idx):
        """
        The training step defines the train loop. It performs the basic functions of an autoencoder, where
        loss is calculated by taking the Mean Squared Error of input against output. This call is independent
        from forward.
        """
        batch_size = len(train_batch)
        # Data must be converted into a tensor format for input into the training sequence.
        x = torch.zeros(batch_size, INPUT_SIZE)
        for index, row in enumerate(train_batch):
            tokenized = tokenize(row)
            x[index, :len(tokenized)] = tokenized
        z = self.encoder(x)
        x_hat = self.decoder(z)
        loss = nn.functional.mse_loss(x_hat, x)
        # Logging to TensorBoard (if installed) by default.
        self.log("train_loss", loss)
        return loss


def tokenize(sequence):
    """
    This helper function translates the string input into a sequence of tokens with a max value of 512.
    Values must be converted into tensor floats in preparation for encoding. The tokenizer model
    in use is the `Bert Base Cased` pre-trained model.
    """
    # This tokenizer auto-truncates to a value of 512.
    tokens = tokenizer.tokenize(sequence, truncation=True)
    ids = tokenizer.convert_tokens_to_ids(tokens)
    return torch.tensor(ids).to(torch.float)


def format_data(raw_bytes):
    """
    The data comes in as a large parquet_file that must be read into a buffer and parsed
    into the correct parquet format.
    """
    reader = pa.BufferReader(raw_bytes)
    table = pq.ParquetFile(reader)
    return table


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
    # config = df_iter.Config(prefix="fineweb/sample/10BT/014_00000.parquet")
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
    trainer = pl.Trainer(limit_train_batches=1, max_epochs=args.epochs)
    trainer.fit(model=my_model, train_dataloaders=data_loader)


if __name__ == "__main__":
    main()
