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

import multiprocessing
import pickle
import unittest
from unittest import mock

from google.cloud import storage

from dataflux_client_python.dataflux_core.tests import fake_gcs
from dataflux_pytorch import dataflux_mapstyle_dataset


class ListingTestCase(unittest.TestCase):

    def setUp(self):
        super().setUp()
        self.project_name = "foo"
        self.bucket_name = "bar"
        self.config = dataflux_mapstyle_dataset.Config(
            num_processes=3,
            max_listing_retries=3,
            prefix="",
            sort_listing_results=True)
        self.data_format_fn = lambda data: data
        client = fake_gcs.Client()

        self.want_objects = [("objectA", 1), ("objectB", 2)]
        for (name, length) in self.want_objects:
            client.bucket(self.bucket_name)._add_file(
                self.config.prefix + name, length * "0")
        client._set_perm([
            dataflux_mapstyle_dataset.CREATE, dataflux_mapstyle_dataset.DELETE
        ], self.bucket_name)
        self.storage_client = client

    @mock.patch("dataflux_pytorch.dataflux_mapstyle_dataset.dataflux_core")
    def test_init(self, mock_dataflux_core):
        """Tests the DataFluxMapStyleDataset can be initiated with the expected listing results."""
        # Arrange.
        mock_listing_controller = mock.Mock()
        mock_listing_controller.run.return_value = self.want_objects
        mock_dataflux_core.fast_list.ListingController.return_value = (
            mock_listing_controller)

        # Act.
        ds = dataflux_mapstyle_dataset.DataFluxMapStyleDataset(
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

    @mock.patch("dataflux_pytorch.dataflux_mapstyle_dataset.dataflux_core")
    def test_init_with_required_parameters(self, mock_dataflux_core):
        """Tests the DataFluxMapStyleDataset can be initiated with only the required parameters."""
        # Arrange.
        mock_listing_controller = mock.Mock()
        mock_listing_controller.run.return_value = self.want_objects
        mock_dataflux_core.fast_list.ListingController.return_value = (
            mock_listing_controller)

        # Act.
        ds = dataflux_mapstyle_dataset.DataFluxMapStyleDataset(
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

    @mock.patch("dataflux_pytorch.dataflux_mapstyle_dataset.dataflux_core")
    def test_init_without_storage_client(self, mock_dataflux_core):
        """Tests the DataFluxMapStyleDataset can be initiated without storage_client."""
        # Arrange.
        mock_listing_controller = mock.Mock()
        mock_listing_controller.run.return_value = self.want_objects
        mock_dataflux_core.fast_list.ListingController.return_value = (
            mock_listing_controller)

        # Act.
        ds = dataflux_mapstyle_dataset.DataFluxMapStyleDataset(
            project_name=self.project_name,
            bucket_name=self.bucket_name,
            # Ensure the dataset can be constructed without setting storage_client.
            # storage_client=self.storage_client,
        )

        # Assert.
        self.assertEqual(
            ds.objects,
            self.want_objects,
            f"got listed objects {ds.objects}, want {self.want_objects}",
        )

    @mock.patch("dataflux_pytorch.dataflux_mapstyle_dataset.dataflux_core")
    def test_init_without_storage_client_constructed_when_needed(
            self, mock_dataflux_core):
        """Tests the DataFluxMapStyleDataset can be initiated without storage_client."""
        # Arrange.
        mock_listing_controller = mock.Mock()
        mock_listing_controller.run.return_value = self.want_objects
        mock_dataflux_core.fast_list.ListingController.return_value = (
            mock_listing_controller)

        # Act.
        ds = dataflux_mapstyle_dataset.DataFluxMapStyleDataset(
            project_name=self.project_name,
            bucket_name=self.bucket_name,
            # Ensure the dataset can be constructed without setting storage_client.
            # storage_client=self.storage_client,
        )

        # Assert.
        # Ensure that client is constructed when not passes by the user.
        self.assertIsNotNone(
            ds.storage_client,
            "storage_client was unexpectedly constructed on init.")
        # Accessing a dataset item calls download_single.
        self.assertIsNotNone(ds[0])
        self.assertEqual(
            mock_dataflux_core.download.download_single.call_count, 1)
        # Verify download_single was called with a storage_client.
        self.assertIsInstance(
            mock_dataflux_core.download.download_single.mock_calls[0].
            kwargs['storage_client'], storage.Client)

    @mock.patch("dataflux_pytorch.dataflux_mapstyle_dataset.dataflux_core")
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
        ds = dataflux_mapstyle_dataset.DataFluxMapStyleDataset(
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

    @mock.patch("dataflux_pytorch.dataflux_mapstyle_dataset.dataflux_core")
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
            ds = dataflux_mapstyle_dataset.DataFluxMapStyleDataset(
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

    @mock.patch("dataflux_pytorch.dataflux_mapstyle_dataset.dataflux_core")
    def test_len(self, mock_dataflux_core):
        """Tests that the len(dataset) method returns the correct number of listed objects."""
        # Arrange.
        mock_listing_controller = mock.Mock()
        mock_listing_controller.run.return_value = self.want_objects
        mock_dataflux_core.fast_list.ListingController.return_value = (
            mock_listing_controller)

        # Act.
        ds = dataflux_mapstyle_dataset.DataFluxMapStyleDataset(
            project_name=self.project_name,
            bucket_name=self.bucket_name,
            config=self.config,
            data_format_fn=self.data_format_fn,
            storage_client=self.storage_client,
        )

        # Assert.
        self.assertEqual(
            len(ds),
            len(self.want_objects),
            f"got len(ds)={len(ds)}, want {len(self.want_objects)}",
        )

    @mock.patch("dataflux_pytorch.dataflux_mapstyle_dataset.dataflux_core")
    def test_getitem(self, mock_dataflux_core):
        """Tests that the dataset[idx] method returns the correct downloaded object content."""
        # Arrange.
        mock_listing_controller = mock.Mock()
        mock_listing_controller.run.return_value = sorted(self.want_objects)
        mock_dataflux_core.fast_list.ListingController.return_value = (
            mock_listing_controller)
        want_downloaded = bytes("content", "utf-8")
        mock_dataflux_core.download.download_single.return_value = want_downloaded
        want_idx = 0

        # Act.
        ds = dataflux_mapstyle_dataset.DataFluxMapStyleDataset(
            project_name=self.project_name,
            bucket_name=self.bucket_name,
            config=self.config,
            data_format_fn=self.data_format_fn,
            storage_client=self.storage_client,
        )
        got_downloaded = ds[want_idx]

        # Assert.
        self.assertEqual(
            got_downloaded,
            want_downloaded,
        )
        mock_dataflux_core.download.download_single.assert_called_with(
            storage_client=self.storage_client,
            bucket_name=self.bucket_name,
            object_name=self.want_objects[want_idx][0],
            retry_config=dataflux_mapstyle_dataset.MODIFIED_RETRY,
        )

    @mock.patch("dataflux_pytorch.dataflux_mapstyle_dataset.dataflux_core")
    def test_getitems(self, mock_dataflux_core):
        """Tests that the dataset.__getitems__ method returns the list of the correct downloaded object content."""
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
        mock_dataflux_core.download.dataflux_download_threaded.return_value = (
            dataflux_download_return_val)

        def data_format_fn(content):
            return len(content)

        want_downloaded = [
            data_format_fn(bytes_content)
            for bytes_content in dataflux_download_return_val
        ]
        want_indices = [0, 1]

        # Act.
        ds = dataflux_mapstyle_dataset.DataFluxMapStyleDataset(
            project_name=self.project_name,
            bucket_name=self.bucket_name,
            config=self.config,
            data_format_fn=data_format_fn,
            storage_client=self.storage_client,
        )
        got_downloaded = ds.__getitems__(want_indices)

        # Assert.
        self.assertEqual(
            got_downloaded,
            want_downloaded,
        )
        mock_dataflux_core.download.dataflux_download_threaded.assert_called_with(
            project_name=self.project_name,
            bucket_name=self.bucket_name,
            objects=self.want_objects,
            storage_client=self.storage_client,
            dataflux_download_optimization_params=want_optimization_params,
            threads=1,
            retry_config=dataflux_mapstyle_dataset.MODIFIED_RETRY,
        )

    @mock.patch("dataflux_pytorch.dataflux_mapstyle_dataset.dataflux_core")
    def test_init_sets_user_agent(self, mock_dataflux_core):
        """Tests that the init function sets the storage client's user agent."""
        mock_listing_controller = mock.Mock()
        mock_listing_controller.run.return_value = self.want_objects
        mock_dataflux_core.fast_list.ListingController.return_value = (
            mock_listing_controller)

        ds = dataflux_mapstyle_dataset.DataFluxMapStyleDataset(
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

    @mock.patch("dataflux_pytorch.dataflux_mapstyle_dataset.dataflux_core")
    def test_list_GCS_blobs_with_spawn_multiprocess(self, mock_dataflux_core):
        """Tests the _list_GCS_blobs_with_retry initializes client before calling dataflux_core.fast_list.ListingController when multiprcessing start method is spawn."""

        # Arrange.
        mock_listing_controller = mock.Mock()
        mock_listing_controller.client = None
        mock_listing_controller.run.return_value = self.want_objects
        mock_dataflux_core.fast_list.ListingController.return_value = (
            mock_listing_controller)

        # Act.
        ds = dataflux_mapstyle_dataset.DataFluxMapStyleDataset(
            project_name=self.project_name,
            bucket_name=self.bucket_name,
            config=self.config,
            data_format_fn=self.data_format_fn,
            storage_client=None,
        )
        # Remove client created by DataFluxMapStyleDataset.
        ds.storage_client = None
        mock_listing_controller.client = None
        ds._list_GCS_blobs_with_retry()

        if (multiprocessing.get_start_method(allow_none=False)
                != dataflux_mapstyle_dataset.FORK):
            self.assertEqual(
                mock_listing_controller.client,
                None,
                f"got client for fast_list{ds.config.max_composite_object_size}, want None",
            )

    def test_init_without_perm(self):
        """Tests that the DataFluxIterableDataset returns permission error when create and delete permissions are missing."""
        # Arrange.
        client = self.storage_client
        client._set_perm([], self.bucket_name)

        # Since required permission is missing, max_composite_object_size is 0.
        with self.assertRaises(PermissionError):
            ds = dataflux_mapstyle_dataset.DataFluxMapStyleDataset(
                project_name=self.project_name,
                bucket_name=self.bucket_name,
                config=self.config,
                data_format_fn=self.data_format_fn,
                storage_client=client,
            )

    @mock.patch("dataflux_pytorch.dataflux_mapstyle_dataset.dataflux_core")
    def test_init_with_perm(self, mock_dataflux_core):
        """Tests that the compose download is not disabled when create and delete permissions exists."""
        # Arrange.
        # Arrange.
        mock_listing_controller = mock.Mock()
        mock_listing_controller.run.return_value = self.want_objects
        mock_dataflux_core.fast_list.ListingController.return_value = (
            mock_listing_controller)
        want_size = self.config.max_composite_object_size

        # Act.
        ds = dataflux_mapstyle_dataset.DataFluxMapStyleDataset(
            project_name=self.project_name,
            bucket_name=self.bucket_name,
            config=self.config,
            data_format_fn=self.data_format_fn,
            storage_client=self.storage_client,
        )

        # Since required permission exists, max_composite_object_size will not change.
        self.assertEqual(
            ds.config.max_composite_object_size,
            want_size,
            f"got max_composite_object_size for compose download{ds.config.max_composite_object_size}, want {want_size}",
        )

    @mock.patch("dataflux_pytorch.dataflux_mapstyle_dataset.dataflux_core")
    def test_getstate(self, mock_dataflux_core):
        """Tests that the dataset.__getitems__ method returns the list of the correct downloaded object content."""
        # Arrange.
        mock_listing_controller = mock.Mock()
        mock_listing_controller.run.return_value = self.want_objects
        mock_dataflux_core.fast_list.ListingController.return_value = (
            mock_listing_controller)

        # Act.
        ds = dataflux_mapstyle_dataset.DataFluxMapStyleDataset(
            project_name=self.project_name,
            bucket_name=self.bucket_name,
            config=self.config,
            storage_client=self.storage_client,
        )

        want_state = ds.__dict__.copy()
        want_state.pop("storage_client")

        states = ds.__getstate__()

        # Assert.
        self.assertEqual(
            states,
            want_state,
            f"got dataflux_mapstyle_dataset params {states}, want {want_state}",
        )

    @mock.patch("dataflux_pytorch.dataflux_mapstyle_dataset.dataflux_core")
    def test_setstate(self, mock_dataflux_core):
        """Tests that the dataset.__getitems__ method returns the list of the correct downloaded object content."""
        # Arrange.
        mock_listing_controller = mock.Mock()
        mock_listing_controller.run.return_value = self.want_objects
        mock_dataflux_core.fast_list.ListingController.return_value = (
            mock_listing_controller)

        ds = dataflux_mapstyle_dataset.DataFluxMapStyleDataset(
            project_name=self.project_name,
            bucket_name=self.bucket_name,
            config=self.config,
            storage_client=self.storage_client,
        )

        # Remove storage_client from dataflux_mapstyle_dataset instance.
        ds.__dict__.pop("storage_client")
        self.assertNotIn(
            "storage_client", ds.__dict__,
            f"Key 'storage_client' should was not removed from dataflux_mapstyle_dataset instance"
        )
        state = ds.__dict__.copy()

        # Act.
        ds.__setstate__(state)

        # Assert.
        self.assertIsInstance(
            ds.__dict__['storage_client'],
            storage.Client,
            f"Key 'storage_client' should exist in dataflux_mapstyle_dataset instance",
        )


if __name__ == "__main__":
    multiprocessing.set_start_method("spawn")
    unittest.main()
