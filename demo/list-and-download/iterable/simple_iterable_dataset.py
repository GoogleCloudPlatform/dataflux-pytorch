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
import multiprocessing
import time

from google.cloud import storage
from google.cloud.storage import retry
from torch.utils import data

from dataflux_pytorch import dataflux_iterable_dataset


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--project", type=str)
    parser.add_argument("--bucket", type=str)
    parser.add_argument("--prefix", type=str)
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--no-dataflux", type=bool, default=False)
    parser.add_argument("--batch-size", type=int, default=100)
    parser.add_argument("--sleep-per-step", type=float, default=1.3604)
    parser.add_argument("--prefetch-factor", type=int, default=2)
    parser.add_argument("--max-composite-object-size",
                        type=int,
                        default=100000000)
    parser.add_argument("--log-level", type=str, default="ERROR")
    parser.add_argument("--retry-timeout", type=float, default=300.0)
    parser.add_argument("--retry-initial", type=float, default=1.0)
    parser.add_argument("--retry-multiplier", type=float, default=1.2)
    parser.add_argument("--retry-maximum", type=float, default=45.0)
    parser.add_argument("--multiprocessing-start-method",
                        type=str,
                        default=None)
    parser.add_argument("--initialize-storage-client",
                        type=bool,
                        default=False)
    return parser.parse_args()


"""
Sample training loop that utilizes the Dataflux Iterable Dataset, iterates over the given bucket and
counts the number of objects/bytes. For example:

$ python3 -m demo.list-and-download.iterable.simple_iterable_dataset --project=<YOUR_PROJECT> --bucket=<YOUR_BUCKET> --prefix=<YOUR_PREFIX> --epochs=2 --num-workers=8

You can also use the --no-dataflux flag to override the configuration so that listing
is done sequentially and objects are downloaded individually, allowing you to compare
performance numbers from Dataflux to a naive GCS-API implementation without Dataflux's
algorithms.
"""


# Define the data_format_fn to transform the data samples.
# NOTE: Make sure to modify this to fit your data format.
def read_image_modified(content_in_bytes):
    return content_in_bytes


def main():
    args = parse_args()
    logging.basicConfig(level=args.log_level)
    multiprocessing.set_start_method(args.multiprocessing_start_method)
    list_start_time = time.time()
    retry_config = retry.DEFAULT_RETRY.with_timeout(
        args.retry_timeout).with_delay(initial=args.retry_initial,
                                       maximum=args.retry_maximum,
                                       multiplier=args.retry_multiplier)
    config = dataflux_iterable_dataset.Config(
        max_composite_object_size=args.max_composite_object_size,
        list_retry_config=retry_config,
        download_retry_config=retry_config)
    if args.no_dataflux:
        print(
            "Overriding parallelism and composite object configurations to simulate non-dataflux loop"
        )
        config.max_composite_object_size = 0
        config.num_processes = 1
    print(f"Listing started at time {list_start_time}")

    if args.prefix:
        config.prefix = args.prefix

    dataset = dataflux_iterable_dataset.DataFluxIterableDataset(
        project_name=args.project,
        bucket_name=args.bucket,
        config=config,
        data_format_fn=read_image_modified,
        storage_client=storage.Client(
            project=args.project) if args.initialize_storage_client else None,
    )
    list_end_time = time.time()
    print(
        f"Listing discovered {len(dataset.objects)} objects in {list_end_time - list_start_time} seconds."
    )
    data_loader = data.DataLoader(
        dataset=dataset,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        prefetch_factor=args.prefetch_factor,
        persistent_workers=True,
        pin_memory=True,
    )
    training_start_time = time.time()
    print(f"Training started at time {training_start_time}")
    for i in range(args.epochs):
        total_objects = 0
        total_bytes = 0
        epoch_start = time.time()
        last_update = time.time()
        for batch in data_loader:
            # A simple sleep function to simulate the GPU training time.
            if args.sleep_per_step:
                time.sleep(args.sleep_per_step)

            total_objects += len(batch)
            for object_bytes in batch:
                total_bytes += len(object_bytes)
            if time.time() - last_update > 5:
                print(
                    f"Iterated over {total_objects} objects and {total_bytes} bytes so far"
                )
                last_update = time.time()
        epoch_end = time.time()
        print(
            f"Epoch {i} took {epoch_end - epoch_start} seconds to iterate over {total_objects} objects and {total_bytes} bytes."
        )
    training_end_time = time.time()
    print(
        f"All training ({args.epochs} epochs) took {training_end_time - training_start_time} seconds."
    )


if __name__ == "__main__":
    main()
