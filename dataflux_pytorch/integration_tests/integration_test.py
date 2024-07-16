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
import io
import os
import time
import unittest
from math import ceil

import numpy
from torch.utils import data

from dataflux_pytorch import (dataflux_iterable_dataset,
                              dataflux_mapstyle_dataset)


# Define the data_format_fn to transform the data samples.
# NOTE: Make sure to modify this to fit your data format.
def read_image_modified(content_in_bytes):
    return numpy.load(io.BytesIO(content_in_bytes), allow_pickle=True)["x"]


class IntegrationTest(unittest.TestCase):

    def get_config(self):
        config = {}
        # Gather env vars into dictionary.
        config["project"] = os.getenv("PROJECT")
        config["bucket"] = os.getenv("BUCKET")
        config["prefix"] = os.getenv("PREFIX")
        config["num_workers"] = os.getenv("LIST_WORKERS")
        config["expected_file_count"] = os.getenv("FILE_COUNT")
        config["expected_total_size"] = os.getenv("TOTAL_FILE_SIZE")
        config["max_compose_bytes"] = os.getenv("MAX_COMPOSE_BYTES")
        config["list_timeout"] = os.getenv("LIST_TIMEOUT")
        config["download_timeout"] = os.getenv("DOWNLOAD_TIMEOUT")
        config["parallelization"] = os.getenv("PARALLELIZATION")
        config["threads_per_process"] = os.getenv("THREADS_PER_PROCESS")

        # Type convert env vars.
        if config["num_workers"]:
            config["num_workers"] = int(config["num_workers"])
        if config["expected_file_count"]:
            config["expected_file_count"] = int(config["expected_file_count"])
        if config["expected_total_size"]:
            config["expected_total_size"] = int(config["expected_total_size"])
        config["max_compose_bytes"] = (int(config["max_compose_bytes"])
                                       if config["max_compose_bytes"] else
                                       100000000)
        if config["list_timeout"]:
            config["list_timeout"] = float(config["list_timeout"])
        if config["download_timeout"]:
            config["download_timeout"] = float(config["download_timeout"])
        config["parallelization"] = (int(config["parallelization"])
                                     if config["parallelization"] else 1)
        config["threads_per_process"] = (int(config["threads_per_process"]) if
                                         config["threads_per_process"] else 1)

        return config

    def train(self, data_loader, epochs, test_config):
        training_start_time = time.time()
        for i in range(epochs):
            total_bytes = 0
            for batch in data_loader:
                # A simple sleep function to simulate the GPU training time.
                time.sleep(0.1)

                for object_bytes in batch:
                    total_bytes += len(object_bytes)
        training_end_time = time.time()
        training_time = training_end_time - training_start_time
        if (test_config["download_timeout"]
                and training_time > test_config["download_timeout"]):
            raise AssertionError(
                f"Expected download operation to complete in under {test_config['download_timeout']} seconds, but took {training_time} seconds."
            )

    def test_list_and_load_iter(self):
        test_config = self.get_config()
        config = dataflux_iterable_dataset.Config()
        if test_config["prefix"]:
            config.prefix = test_config["prefix"]

        list_start_time = time.time()
        dataset = dataflux_iterable_dataset.DataFluxIterableDataset(
            project_name=test_config["project"],
            bucket_name=test_config["bucket"],
            config=config,
            data_format_fn=read_image_modified,
        )
        list_end_time = time.time()
        listing_time = list_end_time - list_start_time
        if (test_config["expected_file_count"] and len(dataset.objects)
                != test_config["expected_file_count"]):
            raise AssertionError(
                f"Expected {test_config['expected_file_count']} files, but got {len(dataset.objects)}"
            )
        if test_config["list_timeout"] and listing_time > test_config[
                "list_timeout"]:
            raise AssertionError(
                f"Expected list operation to complete in under {test_config['list_timeout']} seconds, but took {listing_time} seconds."
            )
        data_loader = data.DataLoader(
            dataset=dataset,
            batch_size=100,
            num_workers=test_config["parallelization"],
            prefetch_factor=2,
            persistent_workers=True,
            pin_memory=True,
        )
        self.train(data_loader, 1, test_config)

    def test_list_and_load_map(self):
        test_config = self.get_config()
        config = dataflux_mapstyle_dataset.Config()
        if test_config["prefix"]:
            config.prefix = test_config["prefix"]

        list_start_time = time.time()
        dataset = dataflux_mapstyle_dataset.DataFluxMapStyleDataset(
            project_name=test_config["project"],
            bucket_name=test_config["bucket"],
            config=config,
            data_format_fn=read_image_modified,
        )
        list_end_time = time.time()
        listing_time = list_end_time - list_start_time
        if (test_config["expected_file_count"]
                and len(dataset) != test_config["expected_file_count"]):
            raise AssertionError(
                f"Expected {test_config['expected_file_count']} files, but got {len(dataset)}"
            )
        if test_config["list_timeout"] and listing_time > test_config[
                "list_timeout"]:
            raise AssertionError(
                f"Expected list operation to complete in under {test_config['list_timeout']} seconds, but took {listing_time} seconds."
            )
        data_loader = data.DataLoader(
            dataset=dataset,
            batch_size=100,
            num_workers=test_config["parallelization"],
            prefetch_factor=2,
            persistent_workers=True,
            pin_memory=True,
        )
        self.train(data_loader, 2, test_config)
