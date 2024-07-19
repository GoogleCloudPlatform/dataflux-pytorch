import torch
import math
from torch import optim, nn, utils, Tensor
from torch.utils import data
from transformers import AutoTokenizer
from dataflux_pytorch import dataflux_iterable_dataset as df_iter
import lightning.pytorch as pl
import pyarrow as pa
import pyarrow.parquet as pq

# def parse_args():
#     parser = argparse.ArgumentParser()
#     parser.add_argument("--project", type=str)
#     parser.add_argument("--bucket", type=str)
#     parser.add_argument("--prefix", type=str)
#     parser.add_argument("--epochs", type=int, default=10)
#     parser.add_argument("--num-workers", type=int, default=0)
#     parser.add_argument("--no-dataflux", type=bool, default=False)
#     parser.add_argument("--batch-size", type=int, default=100)
#     parser.add_argument("--sleep-per-step", type=float, default=1.3604)
#     parser.add_argument("--prefetch-factor", type=int, default=2)
#     parser.add_argument("--threads-per-worker", type=int, default=2)
#     parser.add_argument("--max-composite-object-size",
#                         type=int,
#                         default=100000000)
#     parser.add_argument("--log-level", type=str, default="ERROR")
#     parser.add_argument("--retry-timeout", type=float, default=300.0)
#     parser.add_argument("--retry-initial", type=float, default=1.0)
#     parser.add_argument("--retry-multiplier", type=float, default=1.2)
#     parser.add_argument("--retry-maximum", type=float, default=45.0)
#     return parser.parse_args()

# define any number of nn.Modules (or use your current ones)
encoder = nn.Sequential(nn.Linear(25, 20), nn.ReLU(), nn.Linear(20, 10))
decoder = nn.Sequential(nn.Linear(10, 20), nn.ReLU(), nn.Linear(20, 25))
cols = [
    'text', 'id', 'dump', 'url', 'date', 'file_path', 'language',
    'language_score', 'token_count'
]
tokenizer = AutoTokenizer.from_pretrained("bert-base-cased")


class TextDemoModel(pl.LightningModule):

    def __init__(self, encoder, decoder):
        super().__init__()
        self.encoder = encoder
        self.decoder = decoder

    def configure_optimizers(self):
        optimizer = optim.Adam(self.parameters(), lr=1e-3)
        return optimizer

    def training_step(self, train_batch, batch_idx):
        # training_step defines the train loop.
        # it is independent of forward
        #values_list = [float(len(val)) for val in train_batch.values()]
        tokens = [tokenize(text) for text in train_batch["text"]]
        x = torch.tensor(tokens)
        #x, _ = train_batch
        x = x.view(x.size(0), -1)
        z = self.encoder(x)
        x_hat = self.decoder(z)
        loss = nn.functional.mse_loss(x_hat, x)
        # Logging to TensorBoard (if installed) by default
        # self.log("train_loss", loss)
        return loss


def tokenize(sequence):
    tokens = tokenizer.tokenize(sequence, truncation=True)
    ids = tokenizer.convert_tokens_to_ids(tokens)
    ids = [float(x) for x in ids]
    return ids[:25]


def format_data(parquet_file):
    """
    The data comes in as a large parquet_file that must be read into a buffer and parsed
    into the correct parquet format for processing.
    """
    reader = pa.BufferReader(parquet_file)
    #table = pq.read_table(reader)
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
        An extension of the DataFluxIterableDataset that allows for iteration through subcomponents of
        a downloaded Parquet file.
        """

    def __iter__(self):
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
    config = df_iter.Config(prefix="fineweb/sample/10BT/014_00000.parquet")
    my_model = TextDemoModel(encoder, decoder)

    dataset = ParquetIterableDataset(
        columns=cols,
        batch_size=9,
        project_name="gcs-tess",
        bucket_name="mirvine-benchmark-central1",
        config=config,
        data_format_fn=format_data,
    )
    data_loader = data.DataLoader(
        dataset=dataset,
        batch_size=9,
        num_workers=4,
        persistent_workers=True,
        pin_memory=True,
    )
    trainer = pl.Trainer(limit_train_batches=2, max_epochs=5)
    trainer.fit(model=my_model, train_dataloaders=data_loader)


if __name__ == "__main__":
    main()
