import math
import os
import sys

file_dir = os.path.dirname(__file__)
sys.path.append(file_dir + "/maxtext/MaxText")

import datetime
import random
import time
from types import MappingProxyType
from typing import Sequence, Type

import jax
import pyarrow as pa
import pyarrow.parquet as pq
from absl import app
from maxtext.MaxText import max_logging, pyconfig, storage_utils, train
from torch.utils import data
from torch.utils.data import DataLoader, IterableDataset

from dataflux_pytorch import dataflux_iterable_dataset

TOTAL_TRAINING_TIME_DIRECTORY = "total_training_time"
PER_STEP_DATA_LOADING_TIME_DIRECTORY = "per_step_data_loading_time"
PER_STEP_TIME_DIRECTORY = "per_step_time"
PER_EPOCH_TIME_DIRECTORY = "per_epoch_time"

STEP_BARRIER_MSG = "Synchronize all processes within a step"


def split_list(lst, n):
    """Splits a list into roughly equal sized sublists and pads.

  Args:
    lst: The list to split.
    n: The desired number of sublists.

  Returns:
    A list of sublists.
  """
    # Calculate the size of each sublist.
    size = len(lst) // n

    # Create the sublists.
    sublists = [lst[i * size:(i + 1) * size] for i in range(n)]

    last_idx = n * size

    if last_idx >= len(lst):
        return sublists

    remainder = len(lst) - last_idx

    for i in range(remainder):
        sublists[i].append(lst[last_idx + i])

    # Padding to make sure all nodes are loading the same amount of
    # files. Needed to make sure no deadlocking when the workload
    # is distributed unevenly.
    max_length = max([len(each) for each in sublists])
    for each in sublists:
        while len(each) < max_length:
            each.append(random.choice(lst))

    return sublists


def list_files_walk(start_path='.'):
    dataset_files = []
    for root, _, files in os.walk(start_path):
        for file in files:
            dataset_files.append(os.path.join(root, file))
    return sorted(dataset_files)


def parquet_data_loader(config):
    batch_size = config.local_batch_size
    worker_id = jax.process_index()

    dataset = ParquetIterableDataset(batch_size=batch_size,
                                     columns=["outputs", "image_base64_str"],
                                     rank=worker_id,
                                     world_size=jax.process_count(),
                                     project_name=os.environ["PROJECT"],
                                     bucket_name=os.environ["BUCKET"],
                                     config=dataflux_iterable_dataset.Config(
                                         num_processes=1,
                                         sort_listing_results=True,
                                         prefix=os.environ["PREFIX"]))
    data_loader = DataLoader(
        dataset=dataset,
        num_workers=config.data_loader_num_workers,
        batch_size=batch_size,
        prefetch_factor=config.prefetch_factor,
    )
    return data_loader


def step_barrier_wait(msg, step_count):
    barrier_start = datetime.datetime.now()
    max_logging.log(
        f"STANDALONE DATALOADER : Barrier on host {jax.process_index()} started on step {step_count} at {barrier_start}"
    )

    jax.experimental.multihost_utils.sync_global_devices(msg)

    barrier_end = datetime.datetime.now()
    max_logging.log(
        f"STANDALONE DATALOADER : Barrier on host {jax.process_index()} completed on step {step_count} at {barrier_end}, lasted {(barrier_end - barrier_start).total_seconds()} seconds"
    )


def measure_epoch_time(epoch, epoch_start, epoch_times):
    epoch_end = datetime.datetime.now()
    max_logging.log(
        f"STANDALONE DATALOADER : Host {jax.process_index()} completed epoch {epoch} using {(epoch_end - epoch_start).total_seconds()} seconds"
    )
    epoch_times.append([
        jax.process_index(), epoch, (epoch_end - epoch_start).total_seconds()
    ])


def data_load_loop(config):
    """Main data loader loop.
  Loads batches of data for each training step.
  """
    # We seem to need the mesh to be setup for the distributed barrier to work.
    # Therefore, simply calling this function but using our own iterator.
    train.setup_mesh_and_model(config)
    data_loader = parquet_data_loader(config)

    max_steps = config.max_steps

    # Record per-step per-epoch data loading times.
    per_step_data_loading_time = []

    # Record per-step times, which includes data loading, simulated computation time, and barrier wait time.
    step_time = []

    # Record per-epoch total time.
    epoch_times = []

    jax.experimental.multihost_utils.sync_global_devices(
        "Barrier before training steps start")
    training_start = datetime.datetime.now()
    max_logging.log(
        f"STANDALONE DATALOADER : Started training loop on host {jax.process_index()}"
    )
    global_steps = 0
    for i in range(config.epochs):
        epoch_start = datetime.datetime.now()
        local_steps = 0
        step_data_loading_start = datetime.datetime.now()
        step_start = datetime.datetime.now()
        for _ in data_loader:
            step_data_loading_end = datetime.datetime.now()
            data_loading_interval = (step_data_loading_end -
                                     step_data_loading_start).total_seconds()
            max_logging.log(
                f"STANDALONE DATALOADER : Host {jax.process_index()} got a batch in {data_loading_interval} seconds on epoch {i} step {local_steps}"
            )
            per_step_data_loading_time.append(
                [jax.process_index(), i, local_steps, data_loading_interval])

            if jax.process_index() == 0:
                if data_loading_interval < config.per_step_interval:
                    time.sleep(config.per_step_interval -
                               data_loading_interval)
                step_barrier_wait(STEP_BARRIER_MSG, local_steps)
            else:
                step_barrier_wait(STEP_BARRIER_MSG, local_steps)

            # Measure the per-step time.
            step_end = datetime.datetime.now()
            step_time.append([
                jax.process_index(), i, local_steps,
                (step_end - step_start).total_seconds()
            ])
            step_start = step_end
            # Reset the start time of computing data loading.
            step_data_loading_start = datetime.datetime.now()

            local_steps += 1
            global_steps += 1
            if max_steps > 0 and global_steps >= max_steps:
                max_logging.log(
                    f"STANDALONE DATALOADER : {global_steps} global steps reached. Stopped training."
                )
                break
        else:
            # Only executed if the inner loop did NOT break.
            measure_epoch_time(epoch=i,
                               epoch_start=epoch_start,
                               epoch_times=epoch_times)
            continue
        # Although the training has reached the global steps, we'd still want to
        # log the current epoch time.
        measure_epoch_time(epoch=i,
                           epoch_start=epoch_start,
                           epoch_times=epoch_times)
        break

    training_end = datetime.datetime.now()
    training_time = (training_end - training_start).total_seconds()

    max_logging.log(
        f"STANDALONE DATALOADER Metrics: Training completed in {training_time} seconds, on host {jax.process_index()}"
    )
    max_logging.log(
        f"STANDALONE DATALOADER Metrics: Per-epoch total times on host {jax.process_index()}: {epoch_times}."
    )
    max_logging.log(
        f"STANDALONE DATALOADER Metrics: Per-step data loading times on host {jax.process_index()}: {per_step_data_loading_time}."
    )
    max_logging.log(
        f"STANDALONE DATALOADER Metrics: Per-step times on host {jax.process_index()}: {step_time}.",
    )

    if config.gcs_metrics_bucket:
        max_logging.log(
            f"Uploading metrics to GCS bucket {config.gcs_metrics_bucket} on host {jax.process_index()}"
        )
        base_name = f"{jax.process_index()}.csv"
        # Upload total training time.
        storage_utils.upload_csv(
            config.gcs_metrics_bucket,
            os.path.join(config.run_name, TOTAL_TRAINING_TIME_DIRECTORY,
                         base_name), [[jax.process_index(), training_time]])

        # Upload per-step data loading time.
        storage_utils.upload_csv(
            config.gcs_metrics_bucket,
            os.path.join(config.run_name, PER_STEP_DATA_LOADING_TIME_DIRECTORY,
                         base_name), per_step_data_loading_time)

        # Upload per-step total time.
        storage_utils.upload_csv(
            config.gcs_metrics_bucket,
            os.path.join(config.run_name, PER_STEP_TIME_DIRECTORY, base_name),
            step_time)

        # Upload per epoch time.
        storage_utils.upload_csv(
            config.gcs_metrics_bucket,
            os.path.join(config.run_name, PER_EPOCH_TIME_DIRECTORY, base_name),
            epoch_times)

        max_logging.log(
            f"Finished uploading metrics to GCS bucket {config.gcs_metrics_bucket} on host {jax.process_index()}"
        )
        max_logging.log(f"Run name for accessing metrics: {config.run_name}")


def main(argv: Sequence[str]) -> None:
    jax.config.update("jax_cpu_enable_gloo_collectives", True)
    os.environ["TF_CPP_MIN_LOG_LEVEL"] = "0"
    pyconfig.initialize(argv)
    config = pyconfig.config
    # validate_train_config(config)
    max_logging.log(f"Found {jax.device_count()} devices.")
    max_logging.log(f"Found {jax.process_count()} processes.")
    max_logging.log(f"Found {jax.devices()} devices.")
    if config.dataset_type == "tfds":
        os.environ["TFDS_DATA_DIR"] = config.dataset_path
    data_load_loop(config)


class ParquetIterableDataset(dataflux_iterable_dataset.DataFluxIterableDataset
                             ):

    def __init__(self, columns, batch_size, rank, world_size, *args, **kwargs):
        super().__init__(*args, data_format_fn=self.data_format_fn, **kwargs)
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

    def data_format_fn(self, raw_bytes):
        """
        The data comes in as a large parquet_file that must be read into a buffer and parsed
        into the correct parquet format.
        """
        reader = pa.BufferReader(raw_bytes)
        table = pq.ParquetFile(reader)
        return table

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
            for bytes_content in dataflux_iterable_dataset.dataflux_core.download.dataflux_download_lazy(
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
            max_logging.log(
                f"-----> Worker {self.rank}.{worker_id} downloading {self.objects[start:end]}\n"
            )
            for bytes_content in dataflux_iterable_dataset.dataflux_core.download.dataflux_download_lazy(
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


if __name__ == "__main__":
    app.run(main)
