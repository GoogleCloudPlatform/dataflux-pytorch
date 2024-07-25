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

import math
import unittest
from unittest import mock

from dataflux_client_python.dataflux_core.tests import fake_gcs
from dataflux_pytorch import dataflux_iterable_dataset


class IterableDatasetTestCase(unittest.TestCase):

    def setUp(self):
        super().setUp()
        self.project_name = "foo"
        self.bucket_name = "bar"
        self.config = dataflux_iterable_dataset.Config(
            num_processes=3,
            max_listing_retries=3,
            prefix="",
            sort_listing_results=True,
        )
        self.data_format_fn = lambda data: data
        client = fake_gcs.Client()

        self.want_objects = [("objectA", 1), ("objectB", 2)]
        for (name, length) in self.want_objects:
            client.bucket(self.bucket_name)._add_file(name, length * '0')
        self.storage_client = client

    @mock.patch("dataflux_pytorch.dataflux_iterable_dataset.dataflux_core")
    def test_init(self, mock_dataflux_core):
        """Tests the DataFluxIterableDataset can be initiated with the expected listing results."""
        # Arrange.
        mock_listing_controller = mock.Mock()
        mock_listing_controller.run.return_value = self.want_objects
        mock_dataflux_core.fast_list.ListingController.return_value = (
            mock_listing_controller)

        # Act.
        ds = dataflux_iterable_dataset.DataFluxIterableDataset(
            project_name=self.project_name,
            bucket_name=self.bucket_name,
            config=self.config,
            data_format_fn=self.data_format_fn,
            storage_client=self.storage_client,
        )

        # Assert.
        self.assertEqual(
            ds.objects,
            self.want_objects,
            f"got listed objects {ds.objects}, want {self.want_objects}",
        )

    @mock.patch("dataflux_pytorch.dataflux_iterable_dataset.dataflux_core")
    def test_init_with_required_parameters(self, mock_dataflux_core):
        """Tests the DataFluxIterableDataset can be initiated with only the required parameters."""
        # Arrange.
        mock_listing_controller = mock.Mock()
        mock_listing_controller.run.return_value = self.want_objects
        mock_dataflux_core.fast_list.ListingController.return_value = (
            mock_listing_controller)

        # Act.
        ds = dataflux_iterable_dataset.DataFluxIterableDataset(
            project_name=self.project_name,
            bucket_name=self.bucket_name,
            # storage_client is optional param but still needed here
            # to avoid actual storage.Client construction.
            storage_client=self.storage_client,
        )

        # Assert.
        self.assertEqual(
            ds.objects,
            self.want_objects,
            f"got listed objects {ds.objects}, want {self.want_objects}",
        )

    @mock.patch("dataflux_pytorch.dataflux_iterable_dataset.dataflux_core")
    def test_init_retry_exception_passes(self, mock_dataflux_core):
        """Tests that the initialization retries objects llisting upon exception and passes."""
        # Arrange.
        mock_listing_controller = mock.Mock()

        # Simulate that the first invocation raises an exception and the second invocation
        # passes with the wanted results.
        mock_listing_controller.run.side_effect = [
            Exception(),
            self.want_objects,
            Exception(),
        ]
        mock_dataflux_core.fast_list.ListingController.return_value = (
            mock_listing_controller)

        # Act.
        ds = dataflux_iterable_dataset.DataFluxIterableDataset(
            project_name=self.project_name,
            bucket_name=self.bucket_name,
            config=self.config,
            data_format_fn=self.data_format_fn,
            storage_client=self.storage_client,
        )

        # Assert.
        self.assertEqual(
            ds.objects,
            self.want_objects,
            f"got listed objects {ds.objects}, want {self.want_objects}",
        )

    @mock.patch("dataflux_pytorch.dataflux_iterable_dataset.dataflux_core")
    def test_init_raises_exception_when_retries_exhaust(
            self, mock_dataflux_core):
        """Tests that the initialization raises exception upon exhaustive retries."""
        # Arrange.
        mock_listing_controller = mock.Mock()
        want_exception = RuntimeError("123")

        # Simulate that all retries return with exceptions.
        mock_listing_controller.run.side_effect = [
            want_exception for _ in range(self.config.max_listing_retries)
        ]
        mock_dataflux_core.fast_list.ListingController.return_value = (
            mock_listing_controller)

        # Act & Assert.
        with self.assertRaises(RuntimeError) as re:
            ds = dataflux_iterable_dataset.DataFluxIterableDataset(
                project_name=self.project_name,
                bucket_name=self.bucket_name,
                config=self.config,
                data_format_fn=self.data_format_fn,
                storage_client=self.storage_client,
            )
            self.assertIsNone(
                ds.objects,
                f"got a non-None objects instance variable, want None when all listing retries are exhausted",
            )

        self.assertEqual(
            re.exception,
            want_exception,
            f"got exception {re.exception}, want {want_exception}",
        )

    @mock.patch("dataflux_pytorch.dataflux_iterable_dataset.dataflux_core")
    @mock.patch("torch.utils.data.get_worker_info")
    def test_iter_single_process(self, mock_worker_info, mock_dataflux_core):
        """Tests that the using the iterator of the dataset downloads the list of the correct objects with a single process setup."""
        # Arrange.
        mock_listing_controller = mock.Mock()
        mock_listing_controller.run.return_value = self.want_objects
        mock_dataflux_core.fast_list.ListingController.return_value = (
            mock_listing_controller)
        want_optimization_params = object()
        mock_dataflux_core.download.DataFluxDownloadOptimizationParams.return_value = (
            want_optimization_params)
        dataflux_download_return_val = [
            bytes("contentA", "utf-8"),
            bytes("contentBB", "utf-8"),
        ]

        mock_dataflux_core.download.dataflux_download_lazy.return_value = iter(
            dataflux_download_return_val)
        mock_worker_info.return_value = None

        def data_format_fn(content):
            return len(content)

        want_downloaded = [
            data_format_fn(bytes_content)
            for bytes_content in dataflux_download_return_val
        ]

        # Act.
        ds = dataflux_iterable_dataset.DataFluxIterableDataset(
            project_name=self.project_name,
            bucket_name=self.bucket_name,
            config=self.config,
            data_format_fn=data_format_fn,
            storage_client=self.storage_client,
        )
        got_downloaded = []
        for downloaded in ds:
            got_downloaded.append(downloaded)

        # Assert.
        self.assertEqual(
            got_downloaded,
            want_downloaded,
        )
        # Since this is a single process setup, we expect dataflux_download_lazy to be
        # called with the full list of objects.
        mock_dataflux_core.download.dataflux_download_lazy.assert_called_with(
            project_name=self.project_name,
            bucket_name=self.bucket_name,
            objects=self.want_objects,
            storage_client=self.storage_client,
            dataflux_download_optimization_params=want_optimization_params,
            retry_config=dataflux_iterable_dataset.MODIFIED_RETRY,
        )

    @mock.patch("dataflux_pytorch.dataflux_iterable_dataset.dataflux_core")
    @mock.patch("torch.utils.data.get_worker_info")
    def test_iter_multiple_processes(self, mock_worker_info,
                                     mock_dataflux_core):
        """
        Tests that the using the iterator of the dataset downloads the list of the correct objects with a multi-process setup.
        Specifically, each worker should be assigned to download a different batch of the dataset.
        """
        # Arrange.
        want_objects = [("objectA", 1), ("objectB", 2), ("objectC", 3),
                        ("objectD", 4)]
        mock_listing_controller = mock.Mock()
        mock_listing_controller.run.return_value = want_objects
        mock_dataflux_core.fast_list.ListingController.return_value = (
            mock_listing_controller)
        want_optimization_params = object()
        mock_dataflux_core.download.DataFluxDownloadOptimizationParams.return_value = (
            want_optimization_params)
        dataflux_download_return_val = [
            bytes("contentA", "utf-8"),
            bytes("contentBB", "utf-8"),
        ]

        mock_dataflux_core.download.dataflux_download_lazy.return_value = iter(
            dataflux_download_return_val)

        class _WorkerInfo:
            """A fake WorkerInfo class for testing purpose."""

            def __init__(self, num_workers, id):
                self.num_workers = num_workers
                self.id = id

        num_workers = 2
        id = 0
        want_per_worker = math.ceil(len(want_objects) / num_workers)
        want_start = id * want_per_worker
        want_end = want_start + want_per_worker
        worker_info = _WorkerInfo(num_workers=num_workers, id=id)
        mock_worker_info.return_value = worker_info

        def data_format_fn(content):
            return len(content)

        want_downloaded = [
            data_format_fn(bytes_content)
            for bytes_content in dataflux_download_return_val
        ]

        # Act.
        ds = dataflux_iterable_dataset.DataFluxIterableDataset(
            project_name=self.project_name,
            bucket_name=self.bucket_name,
            config=self.config,
            data_format_fn=data_format_fn,
            storage_client=self.storage_client,
        )
        got_downloaded = []
        for downloaded in ds:
            got_downloaded.append(downloaded)

        # Assert.
        self.assertEqual(
            got_downloaded,
            want_downloaded,
        )
        # Since this is a multi-process setup, we expect dataflux_download_lazy to be
        # only called to download a slice of the objects want_objects[want_start:want_end].
        mock_dataflux_core.download.dataflux_download_lazy.assert_called_with(
            project_name=self.project_name,
            bucket_name=self.bucket_name,
            objects=want_objects[want_start:want_end],
            storage_client=self.storage_client,
            dataflux_download_optimization_params=want_optimization_params,
            retry_config=dataflux_iterable_dataset.MODIFIED_RETRY,
        )

    def test_init_sets_user_agent(self):
        """Tests that the init function sets the storage client's user agent."""
        ds = dataflux_iterable_dataset.DataFluxIterableDataset(
            project_name=self.project_name,
            bucket_name=self.bucket_name,
            config=self.config,
            data_format_fn=self.data_format_fn,
            storage_client=self.storage_client,
        )

        self.assertEqual(
            ds.objects,
            self.want_objects,
            f"got listed objects {ds.objects}, want {self.want_objects}",
        )


if __name__ == "__main__":
    unittest.main()
