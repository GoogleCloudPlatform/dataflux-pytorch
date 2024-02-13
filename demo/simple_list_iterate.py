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
import time

from torch.utils import data
from dataflux_pytorch import dataflux_mapstyle_dataset

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--project", type=str)
    parser.add_argument("--bucket", type=str)
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--no-dataflux", type=bool, default=False)
    parser.add_argument("--batch-size", type=int, default=100)
    parser.add_argument("--prefetch-factor", type=int, default=2)
    return parser.parse_args()

"""
Sample training loop that iterates over the given bucket and counts the number of objects/bytes. For example:

$ python3 -m demo.simple_list_iterate --project=zimbruplayground --bucket=bernardhan-diffusion-small --epochs=2 --num-workers=8

Listing started at time 1706310135.8997579
Listing discovered 10003 objects in 4.665696382522583 seconds.
Training started at time 1706310140.5665798
Iterated over 200 objects and 118331981 bytes so far
Iterated over 1700 objects and 1026198500 bytes so far
Iterated over 3300 objects and 2081709038 bytes so far
Iterated over 4900 objects and 3079001576 bytes so far
Iterated over 6500 objects and 4110702193 bytes so far
Iterated over 8100 objects and 5112547098 bytes so far
Epoch 0 took 38.01222038269043 seconds to iterate over 10003 objects and 6320912111 bytes.
Iterated over 200 objects and 138402866 bytes so far
Iterated over 2100 objects and 1295573582 bytes so far
Iterated over 3400 objects and 2127651395 bytes so far
Iterated over 5300 objects and 3317803831 bytes so far
Iterated over 6200 objects and 3884287582 bytes so far
Iterated over 8400 objects and 5267435784 bytes so far
Iterated over 9400 objects and 5929520073 bytes so far
Epoch 1 took 37.49032020568848 seconds to iterate over 10003 objects and 6320912111 bytes.
All training (2 epochs) took 75.50275588035583 seconds.

You can also use the --no-dataflux flag to override the configuration so that listing
is done sequentially and objects are downloaded individually, allowing you to compare
performance numbers from Dataflux to a naive GCS-API implementation without Dataflux's
algorithms. In this case, all training on the bucket above takes 400 seconds.
"""
def main():
    args = parse_args()
    list_start_time = time.time()
    config = dataflux_mapstyle_dataset.Config()
    if args.no_dataflux:
        print("Overriding parallelism and composite object configurations to simulate non-dataflux loop")
        config.max_composite_object_size = 0
        config.num_processes = 1
    print(f"Listing started at time {list_start_time}")
    dataset = dataflux_mapstyle_dataset.DataFluxMapStyleDataset(args.project, args.bucket, config=config)
    list_end_time = time.time()
    print(f"Listing discovered {len(dataset)} objects in {list_end_time - list_start_time} seconds.")
    data_loader = data.DataLoader(dataset=dataset, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers, prefetch_factor=args.prefetch_factor)
    training_start_time = time.time()
    print(f"Training started at time {training_start_time}")
    for i in range(args.epochs):
        total_objects = 0
        total_bytes = 0
        epoch_start = time.time()
        last_update = time.time()
        for batch in data_loader:
            # Do training here.
            total_objects += len(batch)
            for object_bytes in batch:
                total_bytes += len(object_bytes)
            if time.time() - last_update > 5:
                print(f"Iterated over {total_objects} objects and {total_bytes} bytes so far")
                last_update = time.time()
        epoch_end = time.time()
        print(f"Epoch {i} took {epoch_end - epoch_start} seconds to iterate over {total_objects} objects and {total_bytes} bytes.")
    training_end_time = time.time()
    print(f"All training ({args.epochs} epochs) took {training_end_time - training_start_time} seconds.")

if __name__ == "__main__":
    main()